import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from audiolm_pytorch import AudioLM

from audiolm_pytorch import CoarseTransformer

from trainer import CoarseTransformerTrainer

import yaml
import numpy as np

from tqdm import tqdm
class AudioLMWrapper:
    def __init__(self, config_file, model_path, codec=None):
        
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        coarse_transformer = CoarseTransformer(**config.get("coarse_transformer"))
        coarse_transformer.load(model_path)
        # fine_transformer = FineTransformer(**config.get("fine_transformer"))
        # fine_transformer.load(fine_model_path)

        self.audiolm = AudioLM(
            coarse_transformer = coarse_transformer,
            codec=codec
        ).to("cuda")
    
    def generate(self, artics_features_segments, temperature = 0.1, f0_tokens = None, gt_coarse = None):
        
        
        is_list = True
        if type(artics_features_segments) != type([]):
            artics_features_segments = [artics_features_segments] 
            is_list = False
        
        results = []
        # for artics_features in tqdm(artics_features_segments):
        for artics_features in artics_features_segments:
            if type(artics_features) != type(torch.Tensor()):
                artics_features = torch.from_numpy(artics_features)
            if len(artics_features.shape) == 2:
                artics_features = artics_features.unsqueeze(0)
            out = self.audiolm(artics_features=artics_features, f0_tokens=f0_tokens, temperature = temperature, gt_coarse = gt_coarse)
            # breakpoint()
            # from scipy.io.wavfile import write
            # write("output_audio.wav", 16000, out.T)
            # breakpoint()
            mask = np.any(out != 0, axis=1)
            non_zero_indices = np.where(mask)[0]
            k = non_zero_indices[-1]+1 if non_zero_indices.size > 0 else out.shape[0]
            results.append(out[:k, :])
        
        if is_list:
            return results
        else:
            return results[0]


class AudioLMWrapperOLD:
    def __init__(self, config_file, coarse_model_path, fine_model_path, codec=None):
        
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        coarse_transformer = CoarseTransformer(**config.get("coarse_transformer"))
        coarse_transformer.load(coarse_model_path)
        fine_transformer = FineTransformer(**config.get("fine_transformer"))
        fine_transformer.load(fine_model_path)

        self.audiolm = AudioLM(
            coarse_transformer = coarse_transformer,
            fine_transformer = fine_transformer,
            codec=codec
        ).to("cuda")
    
    def generate(self, artics_features_segments, temperature = 0.01, f0_tokens = None, gt_coarse = None):
        
        
        is_list = True
        if type(artics_features_segments) != type([]):
            artics_features_segments = [artics_features_segments] 
            is_list = False
        
        results = []
        for artics_features in artics_features_segments:
            if type(artics_features) != type(torch.Tensor()):
                artics_features = torch.from_numpy(artics_features)
            if len(artics_features.shape) == 2:
                artics_features = artics_features.unsqueeze(0)
            out = self.audiolm(artics_features=artics_features, f0_tokens=f0_tokens, temperature = temperature, gt_coarse = gt_coarse)
            # from scipy.io.wavfile import write
            # write("output_audio.wav", 16000, out.T)
            # breakpoint()
            mask = np.any(out != 0, axis=1)
            non_zero_indices = np.where(mask)[0]
            k = non_zero_indices[-1] if non_zero_indices.size > 0 else -1
            results.append(out[:k, :])
        
        if is_list:
            return results
        else:
            return results[0]

