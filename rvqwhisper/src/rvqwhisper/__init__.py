from pathlib import Path
from os import PathLike
from typing import BinaryIO, Optional, List
import numpy as np
import yaml

import torch
import torchaudio
from transformers import WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from .model import QuantizedWhisperForConditionalGeneration

MODEL_SAMPLING_RATE = 16_000



class ModelNotFoundError(Exception):
    pass

from vector_quantize_pytorch import ResidualVQ
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper import BatchedInferencePipeline
from einops import rearrange

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
import librosa

class dummyASR:
    def __init__(self):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load model
        # model_name_or_path = "Na0s/Medical-Whisper-Large-v3"
        model_name_or_path = "bofenghuang/whisper-large-v3-distil-multi7-v0.2"
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, torch_dtype=self.torch_dtype)
        self.model.to(self.device)

    def get_transcription(self, audio_path, word_timestamps=False, vad_filter=True, beam_size=5, clip_timestamps = "0", multilingual=True, condition_on_previous_text=False):
        


        array, sr = librosa.load(audio_path, sr=16000)

        # Extract feautres
        input_features = self.processor(
            array, sampling_rate=sr, return_tensors="pt"
        ).input_features


        # Generate tokens
        outputs = self.model.generate(
            input_features.to(self.device, dtype=self.torch_dtype),
            max_new_tokens=128,
            return_token_timestamps=False
        )
        
        # tokens = self.processor.tokenizer.convert_ids_to_tokens(outputs["sequences"][0].tolist())

        # # Get token timestamps as a list (assuming they align with the tokens)
        # token_timestamps = outputs["token_timestamps"][0].tolist()

        # # For debugging, you can look at which token corresponds to which timestamp.
        # token_info = list(zip(tokens, token_timestamps))
        # print("Token info:")
        # for t, ts in token_info:
        #     print(f"Token: {t:10s}  Timestamp: {ts:.2f}")

        # # --- Step 2: Group token-level timestamps into word-level timestamps ---
        # # This example groups tokens into words based on a new word indicator.
        # # Note: Depending on your tokenizer, tokens for a new word might start with a space or a special marker (e.g., "▁").
        # word_segments = []
        # current_word = ""
        # current_start = None

        # for token, ts in token_info:
        #     # Use the tokenizer's convention: if a token starts with a space (or with a marker like "▁"), it indicates a new word.
        #     if token.startswith("Ġ") or token.startswith("▁"):
        #         if current_word:
        #             # End the previous word with the timestamp of the previous token.
        #             word_segments.append({
        #                 "word": current_word,
        #                 "start": current_start,
        #                 "end": prev_ts
        #             })
        #         # Start a new word (strip any leading space/marker)
        #         current_word = token.lstrip(" Ġ")
        #         current_start = ts
        #     else:
        #         # Append subword tokens to the current word.
        #         current_word += token
        #     prev_ts = ts

        # # Append the last accumulated word.
        # if current_word:
        #     word_segments.append({
        #         "word": current_word,
        #         "start": current_start,
        #         "end": prev_ts
        #     })

        # print("\nWord-level timestamps:")
        # for seg in word_segments:
        #     print(seg)
        # breakpoint()

        # Detokenize to text
        # transcription = self.processor.batch_decode(outputs["sequences"], skip_special_tokens=True)[0]
        transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return transcription
    
class RVQFasterWhisperWrapper:
    def __init__(
        self,
        config_path: str | PathLike,
        rvq_model_path_en: Optional[str | PathLike] = None,
        rvq_model_path_fr: Optional[str | PathLike] = None,
        device: str = "cuda",
        model_sampling_rate: int = MODEL_SAMPLING_RATE,
    ):
        self.device = device
        self.model_sampling_rate = model_sampling_rate
        self.config = self.__load_config(config_path)
        self.quantization_config = self.config["quantization"]
        self.processor = self.__get_processor()
        self.rvq_model = {"english": self.__load_rvq_model(rvq_model_path_en)}
        # The French codebooks are optional; the paper's experiments are English only.
        if rvq_model_path_fr and Path(rvq_model_path_fr).exists():
            self.rvq_model["french"] = self.__load_rvq_model(rvq_model_path_fr)
        self.faster_whisper = FasterWhisperModel("large-v2", device=self.device, compute_type="int8_float16")
        # self.inference_pipeline = BatchedInferencePipeline(model=self.faster_whisper)

    def __load_config(self, config_path: str | PathLike) -> dict:
        config_path = Path(config_path)
        if config_path.exists():
            with config_path.open("r") as file:
                config = yaml.safe_load(file)
            return config
        raise FileNotFoundError(
            f"Configuration file does not exists at this path: {str(config_path)}"
        )

    def __get_processor(self):
        model_type = self.config["model_type"]
        return WhisperProcessor.from_pretrained(model_type, task="transcribe")

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
        return vector_quantization.to(self.device).float().eval()

    def __load_trained_model(self, model_path: Optional[str | PathLike]):
        if not model_path:
            model_path = self.config.get("finetuning", {}).get("output_dir", None)
        try:
            model_path = Path(model_path)
            if not model_path.is_file():
                model_path = model_path / "peft_model.pth"
            if not model_path.exists():
                raise ModelNotFoundError(
                    f"Model could not be found at specified location: {str(model_path)}"
                )
        except TypeError:
            raise ModelNotFoundError(
                f"Specified location for model is not valid: {str(model_path)}"
            )
        return torch.load(model_path)

    def __resample_array(self, array: torch.Tensor, input_sampling_rate: int) -> torch.Tensor:
        if input_sampling_rate != self.model_sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=input_sampling_rate, new_freq=self.model_sampling_rate
            )
            return resampler(array)
        return array

    def __get_input_features(self, array: torch.Tensor, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.processor.feature_extractor.hop_length*self.processor.feature_extractor.nb_max_frames

        num_cols = array[0].shape[0]
        output_chunks = []
        # Process chunks
        for i in range(0, num_cols, chunk_size):
            input_features = self.processor(
                array[0][i:i + chunk_size],
                sampling_rate=self.model_sampling_rate,
                return_tensors="pt",
                padding="max_length",
            ).input_features#.to(self.device)
            output_chunks.append(input_features)

        final_output = torch.cat(output_chunks, dim = -1)

        return final_output

    def get_transcription(self, audio_path, word_timestamps=True, vad_filter = True, chunk_length = 30):
        segments, info = self.faster_whisper.transcribe(audio_path, word_timestamps=word_timestamps, vad_filter=vad_filter, chunk_length=chunk_length, beam_size=5, condition_on_previous_text=False, temperature=0.0)
        return list(segments), info
    
    def _decode_frame(self, quantized_indices, lang = "english"):
        if lang.lower() not in self.rvq_model:
            raise ValueError(f"Language {lang} not supported")
        
        batch_size = quantized_indices.shape[0]
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        emb = torch.tensor(0.0, device="cuda")
        for i, indices in enumerate(codes):
            layer = self.rvq_model[lang].layers[i]
            quantized = layer.get_codes_from_indices(indices)
            emb = emb + quantized
    
        bottlenecks = []
        for ind in range(batch_size):
            bottlenecks.append(emb[ind].detach().cpu().numpy())

        return bottlenecks

    def __get_bottleneck(self, input_features, chunk_size=None, vectorize=False, lang="english"):
        if lang.lower() not in self.rvq_model:
            raise ValueError(f"Language {lang} not supported")
        max_len = self.processor.feature_extractor.nb_max_frames
        if chunk_size is None:
            chunk_size = max_len
        target_len = input_features.shape[-1]//2
        num_cols = input_features.shape[-1]
        output_chunks = []
        # Process chunks
        for i in range(0, num_cols, chunk_size):
            chunk = input_features[..., i:i + chunk_size]
            # encoder_outputs = self.model.model.model.encoder(chunk)
            encoder_outputs = torch.asarray(self.faster_whisper.encode(chunk)).to(self.device)
            # quantization_bn, quantization_indices, _, _ = self.model.model.model.vector_quantization(
            #     encoder_outputs.last_hidden_state, return_all_codes = True
            # )
            quantization_bn, quantization_indices, _, _ = self.rvq_model[lang](
                encoder_outputs, return_all_codes = True
            )
            if not vectorize:
                latent_features = torch.as_tensor(quantization_indices).squeeze(0).detach().cpu().numpy()
            else:
                latent_features = torch.as_tensor(quantization_bn).squeeze(0).detach().cpu().numpy()

            output_chunks.append(latent_features)
            
        
        final_output = np.concatenate(output_chunks, axis=0)
        return final_output[...,:target_len]
    
    def __get_forced_decoder_ids(self, tokenizer, language):
        task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")        
        language_token = f"<|{language}|>"
        language_token_id = tokenizer.convert_tokens_to_ids(language_token)
        
        forced_decoder_ids = [
            (1, task_token_id),      
            (2, language_token_id)   
        ]
        
        return forced_decoder_ids

    def decode_from_codebook_indices(self, quantized_indices):
        # Input: batch x num tokens x num quantizers
        # Output: batch x 1 x num samples

        # The following code is hacked in from self.model.decode() (Encodec version 0.1.1) where we skip the part about
        # scaling.
        # Shape: 1 x (num_frames * stride product). 1 because we have 1 frame (because no segmenting)
        frames = self._decode_frame(quantized_indices)
        return frames
    
    def bottleneck_to_text_tokens(self, bottleneck: np.ndarray, language: str = None):
        bottleneck = torch.from_numpy(bottleneck).to(self.device)
        encoder_features = self.model.model.model.vector_quantization.get_output_from_indices(bottleneck)
        encoder_features = encoder_features.unsqueeze(0)
        encoder_outputs = BaseModelOutput(last_hidden_state = encoder_features,
                                          hidden_states = None,
                                          attentions = None)
        forced_decoder_ids = self.__get_forced_decoder_ids(self.processor.tokenizer, language)
        
        text_tokens = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    forced_decoder_ids=forced_decoder_ids
                ).squeeze(0).detach().cpu().numpy()

        return text_tokens

    def compute_text(self, audio_path: BinaryIO | str | PathLike, language: str = None):
        text_tokens = self.compute_text_tokens(audio_path, language)
        self.processor.tokenizer.set_prefix_tokens(language)
        text = self.processor.tokenizer.batch_decode(np.expand_dims(text_tokens, axis=0), skip_special_tokens=True)
        return text

    def compute_text_tokens(self, audio_path: BinaryIO | str | PathLike, language: str = None):
        bottleneck = self.compute_bottleneck(audio_path)
        text_tokens = self.bottleneck_to_text_tokens(bottleneck, language)
        return text_tokens

    def compute_bottleneck_from_array(self, whole_speech_array, file_sampling_rate, vectorize = False, lang = "english"):
        if lang.lower() not in self.rvq_model:
            raise ValueError(f"Language {lang} not supported")
        # whole_speech_array, file_sampling_rate = torchaudio.load(audio_path)
        whole_speech_array = self.__resample_array(whole_speech_array, file_sampling_rate)
        # print("whole_speech_array.shape", whole_speech_array.shape)
        channel_outputs = []
        is_multichannel = False
        if whole_speech_array.shape[0] > 1:
            is_multichannel = True

        for speech_array in whole_speech_array:
            input_features = self.__get_input_features([np.concatenate([speech_array], axis = -1)])
            
            quantization_indices = self.__get_bottleneck(input_features, vectorize=vectorize, lang=lang)
            target_len = speech_array.shape[0]//self.processor.feature_extractor.hop_length//2
            quantization_indices = quantization_indices[:target_len, ...]
            channel_outputs.append(quantization_indices)

        if not is_multichannel:
            return channel_outputs[0]
        else:
            return channel_outputs

    def compute_bottleneck(self, audio_path: BinaryIO | str | PathLike, vectorize = False, lang = "english"):
        if lang.lower() not in self.rvq_model:
            raise ValueError(f"Language {lang} not supported")
        whole_speech_array, file_sampling_rate = torchaudio.load(audio_path)
        whole_speech_array = self.__resample_array(whole_speech_array, file_sampling_rate)
        # print("whole_speech_array.shape", whole_speech_array.shape)
        channel_outputs = []
        is_multichannel = False
        if whole_speech_array.shape[0] > 1:
            is_multichannel = True

        for speech_array in whole_speech_array:
            input_features = self.__get_input_features([np.concatenate([speech_array], axis = -1)])
            
            quantization_indices = self.__get_bottleneck(input_features, vectorize=vectorize, lang=lang)
            target_len = speech_array.shape[0]//self.processor.feature_extractor.hop_length//2
            quantization_indices = quantization_indices[:target_len, ...]
            channel_outputs.append(quantization_indices)

        if not is_multichannel:
            return channel_outputs[0]
        else:
            return channel_outputs

