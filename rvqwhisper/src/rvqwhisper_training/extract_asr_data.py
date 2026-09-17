import argparse
import os
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from transformers import Seq2SeqTrainingArguments

from torch.utils.data import DataLoader
import pickle as pkl

import torch
from load_datasets import (
    get_cv17_asr_dataset,
    get_mls_asr_dataset,
    get_vf_asr_dataset,
    get_vp_asr_dataset,
)
from utils import (
    CustomSeq2SeqTrainer,
    load_config,
    DataCollatorSpeechSeq2SeqWithPadding,
    lang2Lang,
    print_trainable_parameters,
    TensorBoardCallback,
    ValidationCallback,
    CombinedDataset,
    BufferedDataset,
)

from torch.cuda.amp import autocast

import evaluate

import yaml
import torch
import torchaudio

from rvqwhisper.model import QuantizedWhisperForConditionalGeneration
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model


def main(args):
    config = load_config(args.config)
    model_type = config["model_type"]
    quantization_config = config["quantization"]

    model = QuantizedWhisperForConditionalGeneration.from_pretrained(
        model_type,
        vector_quantization_config=quantization_config,
        load_in_8bit=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)  # , output_embedding_layer_name="proj_out")
    loraconfig = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, loraconfig)
    trained_model = torch.load(os.path.join(config["finetuning"]["output_dir"], "peft_model.pth"))
    model.load_state_dict(trained_model["state_dict"], strict=True)
    model = model.to("cuda").half()
    model.eval()

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    language = config["finetuning"]["language"]
    datasets = config["finetuning"]["datasets"]
    streaming = config["finetuning"].get("streaming", False)
    limit = config["finetuning"].get("data_limit", None)
    limit_eval = config["finetuning"].get("eval_data_limit", None)
    data_root = config.get("data_root", None)
    hf_cache = config.get("hf_cache", None)

    tokenizer = WhisperTokenizer.from_pretrained(
        model_type, language=lang2Lang[language], task="transcribe"
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
    processor = WhisperProcessor.from_pretrained(
        model_type, language=lang2Lang[language], task="transcribe"
    )

    list_processed_datasets = []
    list_processed_datasets_test = []
    for dataset in datasets:
        if dataset == "CV17":
            processed_dataset = get_cv17_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                split="train",
                limit=limit,
                cache_dir=hf_cache,
                streaming=streaming,
                padding="max_length",
            )
            processed_dataset_test = get_cv17_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                split="test",
                limit=limit_eval,
                cache_dir=hf_cache,
                streaming=streaming,
                padding="max_length",
            )

        if dataset == "MLS":
            processed_dataset = get_mls_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                split="train",
                limit=limit,
                cache_dir=hf_cache,
                streaming=streaming,
                padding="max_length",
            )
            processed_dataset_test = get_mls_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                split="test",
                limit=limit_eval,
                cache_dir=hf_cache,
                streaming=streaming,
                padding="max_length",
            )

        if dataset == "VF":
            processed_dataset = get_vf_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                data_root=data_root,
                limit=limit,
                split="train",
                padding="max_length",
            )
            processed_dataset_test = get_vf_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                data_root=data_root,
                limit=limit_eval,
                split="test",
                padding="max_length",
            )

        if dataset == "VP":
            processed_dataset = get_vp_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                data_root=data_root,
                split="train",
                limit=limit,
                cache_dir=hf_cache,
                streaming=streaming,
                padding="max_length",
            )
            processed_dataset_test = get_vp_asr_dataset(
                [language],
                [feature_extractor],
                [tokenizer],
                data_root=data_root,
                split="test",
                limit=limit_eval,
                cache_dir=hf_cache,
                streaming=streaming,
                padding="max_length",
            )

        list_processed_datasets.append(processed_dataset[language])
        list_processed_datasets_test.append(processed_dataset_test[language])

    # breakpoint()

    final_dataset_train = CombinedDataset(list_processed_datasets, shuffle=True)
    final_dataset_test = CombinedDataset(list_processed_datasets_test, shuffle=True)
    device = "cuda"
    for sample in final_dataset_train:
        audio_path = sample["path"]
        speech_array, sampling_rate = torchaudio.load(audio_path)
        input_features = processor(
            speech_array[0], sampling_rate=sampling_rate, return_tensors="pt", padding=False
        ).input_features.to(device)
        # breakpoint()
        with autocast():
            genz = model.model.generate(
                input_features,
                return_token_timestamps=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                task="transcribe",
                language="fr",
            )

        encoder_outputs = model.model.model.encoder(input_features.half())
        (
            encoder_outputs_quantized,
            quantization_indices,
            quantization_loss,
            all_encoder_outputs_quantized,
        ) = model.model.model.vector_quantization(
            encoder_outputs.last_hidden_state.half(), return_all_codes=True
        )
        breakpoint()

        print(".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process dataset based on configurations."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--force_compute",
        type=bool,
        default=False,
        help="recompute all steps even if already computed",
    )

    args = parser.parse_args()
    main(args)
