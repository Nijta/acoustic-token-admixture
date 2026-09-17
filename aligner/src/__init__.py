from os import PathLike
import os
from pathlib import Path
from typing import Optional
from einops import rearrange
import torch
from vector_quantize_pytorch import ResidualVQ
import yaml
import numpy as np

try:
    from aligner.src.training import ForcedAligner, CTCHead, Config, Aligner, PhonemeInference, TextDurationPredictor
    from aligner.src.f0predictor import F0Predictor
except:
    from training import ForcedAligner, CTCHead, Config, Aligner, PhonemeInference, TextDurationPredictor
    from f0predictor import F0Predictor
class AlignerWrapper:
    def __init__(self, aligner_model_path, rvq_config_path, rvq_model_path, lang = "eng", f0_predictor_model_path = None, duration_predictor_model_path = None, device = "cuda"):
        self.aligner_model_path = aligner_model_path
        self.config = self.__load_config(rvq_config_path)
        self.rvq_model_path = rvq_model_path
        self.rvq_model = self.__load_rvq_model(rvq_model_path)
        self.config_align = Config()
        self.model = CTCHead(self.config_align.INPUT_DIM, self.config_align.NUM_PHONEMES).to(self.config_align.DEVICE)
        checkpoint = torch.load(aligner_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.aligner = Aligner(lang=lang)
        self.forced_aligner = ForcedAligner(self.model, self.aligner, blank=self.config_align.NUM_PHONEMES)
        self.phoneme_inference = PhonemeInference(self.aligner, aligner_model_path)
        self.f0_predictor_model_path = f0_predictor_model_path
        self.duration_predictor_model_path = duration_predictor_model_path
        self.f0model = F0Predictor().to(self.config_align.DEVICE)
        num_phonemes = len(self.aligner.phone_to_id)
        self.durmodel = TextDurationPredictor(num_phonemes).to(self.config_align.DEVICE)
        if self.f0_predictor_model_path:
            f0_checkpoint = torch.load(self.f0_predictor_model_path)
            self.f0model.load_state_dict(f0_checkpoint)
        if self.duration_predictor_model_path:
            duration_checkpoint = torch.load(self.duration_predictor_model_path)
            self.durmodel.load_state_dict(duration_checkpoint)
        self.f0mean = 151.38564 
        self.f0_std = 50.393913
        self.device = device

    def __load_config(self, config_path: str | PathLike) -> dict:
        config_path = Path(config_path)
        if config_path.exists():
            with config_path.open("r") as file:
                config = yaml.safe_load(file)
            return config
        raise FileNotFoundError(
            f"Configuration file does not exists at this path: {str(config_path)}"
        )
    
    def decode_from_codebook_indices(self, quantized_indices):
        frames = self._decode_frame(quantized_indices)
        return frames
    
    def _decode_frame(self, quantized_indices):
        batch_size = quantized_indices.shape[0]
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        emb = torch.tensor(0.0, device=codes.device)
        for i, indices in enumerate(codes):
            layer = self.rvq_model.layers[i]
            quantized = layer.get_codes_from_indices(indices)
            emb = emb + quantized
    
        bottlenecks = []
        for ind in range(batch_size):
            bottlenecks.append(emb[ind].detach().cpu().numpy())

        return bottlenecks
    
    def __load_rvq_model(self, model_path: Optional[str | PathLike]):
        quantization_config = self.config["quantization"]
        trained_model = self.__load_trained_model(model_path)
        vector_quantization = ResidualVQ(
            dim=quantization_config["dim"],
            codebook_size=quantization_config["codebook_size"],
            num_quantizers=quantization_config["num_quantizers"],
        ).to("cuda")
        # vector_quantization.load_state_dict(trained_model["state_dict"], strict=True)
        vector_quantization.load_state_dict(trained_model, strict=True)
        return vector_quantization.to("cuda").float().eval()
    
    def __load_trained_model(self, model_path: Optional[str | PathLike]):
        if not model_path:
            model_path = self.config.get("finetuning", {}).get("output_dir", None)
        try:
            model_path = Path(model_path)
            if not model_path.is_file():
                model_path = model_path / "peft_model.pth"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model could not be found at specified location: {str(model_path)}"
                )
        except TypeError:
            raise FileNotFoundError(
                f"Specified location for model is not valid: {str(model_path)}"
            )
        return torch.load(model_path)
    


    def sum_segments_between_zeros(self, lst, ids):
        segment_sums = [[0]]
        current_sum = 0

        for dur, id in zip(lst, ids):
            if id == 0:
                segment_sums[-1].append(current_sum)  # Append current_sum even if it's 0
                # current_sum = 0  # Reset for the next segment
                segment_sums.append([current_sum+dur.item()])

            current_sum += dur.item()

        return segment_sums[1:-1]

    def predict_artics(self, transcription, scale_factor = 1.0, return_word_timestamps = False, prefix = "I want to say, ", postfix = ". did you hear me?"):
        
        target_ids = self.aligner.extract_phonemes(prefix+transcription+postfix)
        phoneme_tensor = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0).to(Config.DEVICE)
        ali = self.durmodel(phoneme_tensor).squeeze()
        artics = np.concatenate([np.asarray(self.forced_aligner.phone_to_artics[self.forced_aligner.id_to_phone[x]])[None, ...] if self.forced_aligner.id_to_phone[x] != " " else self.forced_aligner.xos_tensor
                            for x in target_ids], axis=0)
        assert len(ali) == len(artics)
        ali = torch.round(ali * scale_factor).long().detach().cpu()
        repeated_tokens = torch.from_numpy(artics).repeat_interleave(ali, dim=0).float().detach().cpu().numpy()
            
        if not return_word_timestamps:
            if prefix != "" or postfix != "":
                prefix_len = len(self.aligner.extract_phonemes(prefix))-1
                postfix_len = len(self.aligner.extract_phonemes(postfix))-1
                return repeated_tokens, sum(ali[:prefix_len]), sum(ali[-postfix_len:])
            return repeated_tokens
        
        word_sums = self.sum_segments_between_zeros(ali, target_ids)
        return repeated_tokens, word_sums
    
    def predict_f0(self, artics, low_threshold = 20.0):
        artics = torch.from_numpy(artics).to(Config.DEVICE)
        lens =  torch.from_numpy(np.asarray([artics.shape[0]])).to(Config.DEVICE)
        with torch.no_grad():
            predicted_f0 = self.f0model(artics.unsqueeze(0),lens).squeeze().detach().cpu().numpy()
        predicted_f0 = (predicted_f0 * self.f0_std) + self.f0mean 
        predicted_f0[predicted_f0<low_threshold] = 0.0
        return predicted_f0

    def align_bottleneck_with_transcription(self, tokens, transcription, scale_factor = 1.0, return_word_timestamps = False, phoneme_inference = False):
        bn = self.decode_from_codebook_indices(torch.from_numpy(tokens).unsqueeze(0).long())[0]
        # breakpoint()
        # transcription = "Il suffisait qu'on a une visite la, ramène-moi ci, ramène-moi ça. On est ravitaillé aussitôt. C'est soit les familles, soit les amis, soit les patients qui vont à l'extérieur. Ah c'est porte ouverte. ça se limitait à la cocaïne, l'héroïne. Maintenant on a ce qu'on appelle les ecstasys, les buvards, les médicaments. Ils ont beau leur faire la guerre, mais ça continue tout le temps."
        if not phoneme_inference:
            target_ids = self.aligner.extract_phonemes(transcription)
            
            ali, artics = self.forced_aligner.align(
                audio_path=bn,
                target_ids=target_ids,
            )
            assert len(ali) == len(artics)
            ali = torch.round(ali * scale_factor).long()
            # breakpoint()
            repeated_tokens = artics.repeat_interleave(ali, dim=0).float().detach().cpu().numpy()
        else:
            repeated_tokens = self.phoneme_inference.predict(bn)


        
        if not return_word_timestamps:
            return repeated_tokens
        
        if phoneme_inference:
            raise "unable to output word timestaps using phoneme inference"
        
        word_sums = self.sum_segments_between_zeros(ali, target_ids)
        return repeated_tokens, word_sums
    
    def get_articulatory_features(self, tokens, segments, info, scale_factor = 1.0, return_word_timestamps = False, phoneme_inference = False):
        """
        Process audio segments with their transcriptions using in-memory trimming.
        
        Args:
            audio_path: Path to the audio file.
            segments: List of segment dictionaries with 'start', 'end', and 'text' keys.
        
        Returns:
            List of results from processing each segment.
        """
        # Load the entire audio file using librosa (mono by default)
        seq_len = tokens.shape[0]
        results = []
        segments_alignment = []
        if return_word_timestamps:
            word_timestamps = []
        for segment in segments:
            start_sample = round((segment.start/info.duration)*seq_len)
            end_sample = round((segment.end/info.duration)*seq_len)
            text = segment.text.strip()#.replace(".","")
            
            # Ensure samples are within valid range
            start_sample = max(0, start_sample)
            end_sample = min(seq_len, end_sample)
            
            if start_sample >= end_sample:
                # Handle empty or invalid segments
                results.append(None)  # or raise an error/skip
                continue
            
            # Trim the audio segment
            trimmed_tokens = tokens[start_sample:end_sample,...]
            
            result = self.align_bottleneck_with_transcription(trimmed_tokens, text, scale_factor=scale_factor, return_word_timestamps=return_word_timestamps, phoneme_inference=phoneme_inference)
            if return_word_timestamps:
                result, these_timestamps = result     
                these_timestamps = [[t, segment.start + (x[0]-3)/50, segment.start + (x[1]+1.5)/50] for t, x in zip(text.split(), these_timestamps)]
                word_timestamps.extend(these_timestamps)
            
            results.append(result)
            segments_alignment.append((start_sample, end_sample))
        
        if return_word_timestamps:
            return results, segments_alignment, word_timestamps
        else:
            return results, segments_alignment
    
