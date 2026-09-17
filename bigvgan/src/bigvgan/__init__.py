from pathlib import Path
from os import PathLike
from typing import Optional
import numpy as np
import json
import yaml
import torch
from vector_quantize_pytorch import ResidualVQ
from einops import rearrange
from .bigvgan import BigVGAN as Generator

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)
PITCH_MEAN = 61.64
PITCH_STD = 67.5
XVECTOR_MEAN = 20


class ModelNotFoundError(Exception):
    pass


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class BigVGANWrapper:
    def __init__(
        self,
        config_path: str | PathLike,
        model_path: str | PathLike = None,
        rvq_config_path = None,
        rvq_model_path = None,
        device: str = "cuda",
    ):
        self.device = device
        self.config = self.__load_config(config_path)
        self.model = self.__load_model(model_path)

        self.rvq_config = self.__load_rvq_config(rvq_config_path)
        self.rvq_model_path = rvq_model_path
        self.rvq_model = self.__load_rvq_model(rvq_model_path)

    def __load_config(self, config_path: str | PathLike) -> dict:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            return h

        raise FileNotFoundError(
            f"Configuration file does not exists at this path: {str(config_path)}"
        )

    def __load_rvq_config(self, config_path: str | PathLike) -> dict:
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
        emb = torch.tensor(0.0, device="cuda")
        for i, indices in enumerate(codes):
            layer = self.rvq_model.layers[i]
            quantized = layer.get_codes_from_indices(indices)
            emb = emb + quantized
    
        bottlenecks = []
        for ind in range(batch_size):
            bottlenecks.append(emb[ind].detach().cpu().numpy())

        return bottlenecks
    
    def __load_rvq_model(self, model_path: Optional[str | PathLike]):
        quantization_config = self.rvq_config["quantization"]
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
                raise ModelNotFoundError(
                    f"Model could not be found at specified location: {str(model_path)}"
                )
        except TypeError:
            raise ModelNotFoundError(
                f"Specified location for model is not valid: {str(model_path)}"
            )
        return torch.load(model_path)
    
    def __load_model(self, model_path: str | PathLike):
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model could not be found at specified location: {str(model_path)}"
            )
        state_dict_g = torch.load(model_path, map_location=self.device)
        generator = Generator(self.config, use_cuda_kernel=False).to(device=self.device)
        generator.load_state_dict(state_dict_g["generator"])
        generator.eval()
        generator.remove_weight_norm()

        return generator

    def __run_on_chunks(self, x, chunk_size: int = 200, overlap: int = None):
        """
        Process the input x (a tensor of shape [channels, frames]) in overlapping chunks.
        Uses overlap-add with a Hann window in the output (audio) domain.
        """
        # Set default overlap to 50% of the chunk size if not provided.
        if overlap is None:
            overlap = chunk_size // 2

        total_input_frames = x.shape[-1]

        # If the input is short enough, process it in one go.
        if total_input_frames <= chunk_size:
            y = self.model(x.unsqueeze(0))
            y = y.detach().squeeze().cpu().numpy()
            return y * MAX_WAV_VALUE

        # Run a dummy full chunk to determine the output length per chunk.
        dummy_chunk = x[:, :chunk_size]
        dummy_output = self.model(dummy_chunk.unsqueeze(0))
        dummy_output = dummy_output.detach().squeeze().cpu().numpy()
        out_chunk_length = dummy_output.shape[-1]
        # Compute the effective upsampling factor (ratio between output length and input chunk length)
        r = out_chunk_length / chunk_size

        # Compute the step (hop) size in the input domain.
        step = chunk_size - overlap

        # Determine the starting index of the last chunk in the input domain.
        last_start = ((total_input_frames - 1) // step) * step
        # Allocate accumulators in the output (audio) domain.
        accum_length = int(np.ceil(last_start * r + out_chunk_length))
        output_accum = np.zeros(accum_length)
        weight_accum = np.zeros(accum_length)

        # Process each chunk.
        for start in range(0, total_input_frames, step):
            end = start + chunk_size

            # If the chunk goes beyond the input, pad it with zeros.
            if end > total_input_frames:
                x_chunk = x[:, start:total_input_frames]
                pad_length = end - total_input_frames
                pad_tensor = torch.zeros(x.shape[0], pad_length, device=x.device, dtype=x.dtype)
                x_chunk = torch.cat([x_chunk, pad_tensor], dim=-1)
            else:
                x_chunk = x[:, start:end]

            # Process the chunk.
            y_chunk = self.model(x_chunk.unsqueeze(0))
            y_chunk = y_chunk.detach().squeeze().cpu().numpy()
            current_chunk_length = y_chunk.shape[-1]

            # Create a Hann window matching the length of the model output for this chunk.
            window = np.hanning(current_chunk_length)

            # Compute the corresponding start index in the output domain.
            out_start = int(round(start * r))
            out_end = out_start + current_chunk_length

            # Accumulate the windowed output.
            output_accum[out_start:out_end] += y_chunk * window
            weight_accum[out_start:out_end] += window

        # Normalize to avoid artifacts (make sure no division-by-zero occurs).
        weight_accum[weight_accum == 0] = 1.0
        y_out = output_accum / weight_accum

        # Trim the final output to the expected length.
        expected_length = int(round(total_input_frames * r))
        y_out = y_out[:expected_length]

        return y_out * MAX_WAV_VALUE


    def synthesize(
        self, bottleneck: np.ndarray, xvector: np.ndarray, pitch: np.ndarray, chunk_size: int = 200
    ):
        """
        Synthesize audio given bottleneck, xvector, and pitch.
        Applies necessary normalization and interpolation, then processes in chunks.
        """
        # Normalize pitch and xvector.
        pitch = (pitch - PITCH_MEAN) / PITCH_STD
        xvector = xvector / XVECTOR_MEAN

        # Interpolate pitch to match the bottleneck length.
        x_old = np.linspace(0, 1, pitch.shape[0])
        x_new = np.linspace(0, 1, bottleneck.shape[0])
        pitch = np.interp(x_new, x_old, pitch)
        pitch = pitch.reshape(-1, 1)

        # Repeat xvector to match the interpolated pitch length.
        xvector = np.tile(xvector, (pitch.shape[0], 1))
        # Concatenate bottleneck, pitch, and xvector to form the model input.
        mel = np.hstack((bottleneck, pitch, xvector))
        x = torch.from_numpy(mel).float().transpose(0, 1).to(self.device)

        # Process the input in overlapping chunks.
        anonymized_audio = self.__run_on_chunks(x, chunk_size)

        return anonymized_audio.astype("int16"), self.config.sampling_rate



    def compute_windowed_correlation(self, x, y, max_shift=2):
        """
        Compute the correlation between x and y, accounting for small shifts (up to max_shift).

        Parameters:
            x (np.array): Array of shape (t,).
            y (np.array): Array of shape (t,).
            max_shift (int): Maximum allowed shift (default: 3).

        Returns:
            correlations (np.array): Array of shape (t,) containing the maximum correlation for each time step.
        """
        t = len(x)
        correlations = np.zeros(t)

        for i in range(t):
            # Define the window for y
            start = max(0, i - max_shift)
            end = min(t, i + max_shift + 1)

            # Extract the window from y
            y_window = y[start:end]

            # Compute correlation between x[i] and the y window
            corr_values = np.array([np.corrcoef(x[i], y_window[j])[0, 1] for j in range(len(y_window))])

            # Store the maximum correlation value
            correlations[i] = np.max(corr_values)

        return correlations

    def admixture(self, tokens, tokens_syn, admixture_ratio = 0.5, block_hallucination = False):
        
        if 1024 in tokens_syn[:,0]:
            bad = (tokens_syn == 1024) | (tokens_syn == -1)
            cols_to_replace = np.any(bad, axis=0)
            tokens_syn[:, cols_to_replace] = tokens[:, cols_to_replace]
            
        bottleneck = self.decode_from_codebook_indices(tokens[None,...])[0]
        bottleneck_syn = self.decode_from_codebook_indices(tokens_syn[None,...])[0]
        min_d = min(bottleneck.shape[0], bottleneck_syn.shape[0])
        if bottleneck.shape[0] != bottleneck_syn.shape[0]:
            bottleneck_syn = bottleneck_syn[:min_d,...]
            bottleneck = bottleneck[:min_d,...]
        

        # breakpoint()
        # corr = [np.corrcoef(bottleneck[i, :], bottleneck_syn[i, :])[0, 1] for i in range(bottleneck.shape[0])]
        corr = self.compute_windowed_correlation(bottleneck_syn, bottleneck)
        final_bottleneck = np.zeros_like(bottleneck)
        # can be using other strategies (for example phoneme based)
        for i in range(min_d):
            # if i < 36:
            #     final_bottleneck[i,...] = bottleneck_syn[i,...]
            # elif i>= 36 and i < 45:
            #     final_bottleneck[i,...] = bottleneck[i,...]
            # elif i>=45 and i < 57:
            #     final_bottleneck[i,...] = bottleneck[45 + int((i-45)/2),...]
            # elif i>= 57 and i < 90:
            #     final_bottleneck[i,...] = bottleneck[i,...]
            # elif i>= 90 and i < 162:
            #     final_bottleneck[i,...] = bottleneck_syn[i,...]
            # elif i>= 162 and i<549:
            #     final_bottleneck[i,...] = bottleneck[i,...]
            # elif i>= 549 and i<687:
            #     final_bottleneck[i,...] = bottleneck_syn[i,...]
            # elif i>= 687 and i < 780:
            #     final_bottleneck[i,...] = bottleneck[i,...]
            # elif i>= 780 and i<853:
            #     final_bottleneck[i,...] = bottleneck_syn[i,...]
            # elif i>= 853:
            #     final_bottleneck[i,...] = bottleneck[i,...]
            

            if np.random.random() > admixture_ratio or (block_hallucination and corr[i]<0.6):
                final_bottleneck[i,...] = bottleneck[i,...]
            else:
                final_bottleneck[i,...] = bottleneck_syn[i,...]

        return final_bottleneck, corr
    
    def f0_transformation(self, x, n=32, a=0.75, dB=0):
        if a == 0.0 and dB == 0:
            return x
        # Step 1: Identify non-zero indices and extract non-zero values.
        nonzero_indices = np.nonzero(x)[0]
        nonzero_values = x[nonzero_indices]
        
        # Step 2: Compute the n-frame moving average on the non-zero parts.
        # Using convolution with a uniform kernel. Using mode='same' to preserve length.
        actual_n = min(n, len(nonzero_values))
        kernel = np.ones(actual_n) / actual_n
        f_ma = np.convolve(nonzero_values, kernel, mode='same')
        
        # Step 3: Compute f_new = (1 - a) * f + a * f_ma.
        f_new = (1 - a) * nonzero_values + a * f_ma
        if dB == 0:
            result = x.copy()
            result[nonzero_indices] = f_new
            return result
        
        # Step 4: Add AWGN with a dB SNR.
        # Calculate the power of the signal f_new.
        P_signal = np.mean(f_new ** 2)
        # For 10 dB SNR, noise variance = P_signal / (10^(SNR/10)) = P_signal / 10.
        noise_variance = P_signal / dB
        sigma = np.sqrt(noise_variance)
        noise = np.random.normal(0, sigma, size=f_new.shape)
        f_new_noisy = f_new + noise
        f_new_noisy = np.clip(f_new_noisy, 0, None)

        # Step 5: Create an output array and put the processed non-zero values back into their original positions.
        result = x.copy()
        result[nonzero_indices] = f_new_noisy
        
        return result