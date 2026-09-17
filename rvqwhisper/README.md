# RVQWhisper

edit the configuration file: `config.yaml`

### Finetuning

run finetuning process: `python finetuning.py --config config.yaml`

track the process using tensorboard from your output dir in config:
`tensorboard --logdir /path/to/output_dir --host 0.0.0.0`

### Evaluation

run evaluation: `python evaluation.py --config config.yaml`

### Visualization

simply run `python visualization.py` then look at the output path which is hard coded in the visualization.py script.

# Inference

```python
################### imports

import yaml
import torch
import torchaudio

from rvqwhisper.model import QuantizedWhisperForConditionalGeneration
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

################### preparation

with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)
model_type = config["model_type"]
processor = WhisperProcessor.from_pretrained(model_type, language = "en", task = "transcribe")
quantization_config = config['quantization']
model = QuantizedWhisperForConditionalGeneration.from_pretrained(model_type, vector_quantization_config = quantization_config, load_in_8bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)#, output_embedding_layer_name="proj_out")
loraconfig = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, loraconfig)
trained_model = torch.load(os.path.join(config["finetuning"]["output_dir"], "peft_model.pth"))
model.load_state_dict(trained_model["state_dict"], strict=True)
model = model.to("cuda").half()
model.eval()

################# Inference

speech_array, sampling_rate = torchaudio.load(audio_path)
input_features = processor(speech_array[0], sampling_rate=sampling_rate, return_tensors="pt", padding = False).input_features.to(device)
encoder_outputs = model.model.model.encoder(input_features.half())
encoder_outputs_quantized, quantization_indices, quantization_loss, all_encoder_outputs_quantized = model.model.model.vector_quantization(encoder_outputs.last_hidden_state.half(), return_all_codes = True)

```
