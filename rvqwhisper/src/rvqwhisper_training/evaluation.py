import argparse
from functools import partial
import os
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import torch

from rvqwhisper.model import QuantizedWhisperForConditionalGeneration

from torch.cuda.amp import autocast

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

import evaluate


def run_evaluation(model, dataloader, tokenizer):
    restore_training = False
    if model.training:
        model.eval()
        restore_training = True
    metric = evaluate.load("wer")
    for step, batch in enumerate(tqdm(dataloader)):
        with autocast():
            with torch.no_grad():
                # breakpoint()
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255,
                        # language = "french",
                        # task="transcribe",
                    )
                    .cpu()
                    .numpy()
                )

                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # print(f"{decoded_labels=}")
                # print(f"{decoded_preds=}")
                # breakpoint()
                metric.add_batch(
                    predictions=[x.lower() for x in decoded_preds],
                    references=[x.lower() for x in decoded_labels],
                )
    # breakpoint()

    wer = 100 * metric.compute()
    # print(f"{wer=}")
    if restore_training:
        model.train()
    return wer


def main(args):
    from load_datasets import (
        get_cv17_asr_dataset,
        get_mls_asr_dataset,
        get_vf_asr_dataset,
        get_vp_asr_dataset,
    )
    from utils import load_config, DataCollatorSpeechSeq2SeqWithPadding, lang2Lang

    config = load_config(args.config)
    model_type = config["model_type"]
    data_root = config.get("data_root", None)
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
    model = model.to("cuda")

    # model = WhisperForConditionalGeneration.from_pretrained(model_type, load_in_8bit=True, device_map="auto")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.eval()

    datasets = config["evaluation"]["datasets"]
    batch_size = config["evaluation"]["batch_size"]
    limit = config["evaluation"].get("data_limit", None)
    streaming = config["evaluation"].get("streaming", False)

    for dataset in datasets.keys():
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dataset: ", dataset)
        if dataset == "CV17":
            func = partial(get_cv17_asr_dataset, streaming=streaming)
        if dataset == "MLS":
            func = partial(get_mls_asr_dataset, streaming=streaming)
        if dataset == "VF":
            func = partial(get_vf_asr_dataset, data_root=data_root)
        if dataset == "VP":
            func = partial(
                get_vp_asr_dataset, streaming=streaming, cache_dir="data/voxpopuli_fr"
            )

        languages = config["evaluation"]["datasets"][dataset]
        tokenizers = [
            WhisperTokenizer.from_pretrained(
                model_type, language=lang2Lang[lang], task="transcribe"
            )
            for lang in languages
        ]
        feature_extractors = [
            WhisperFeatureExtractor.from_pretrained(model_type) for _ in languages
        ]
        processors = [
            WhisperProcessor.from_pretrained(
                model_type, language=lang2Lang[lang], task="transcribe"
            )
            for lang in languages
        ]
        processed_dataset = func(
            languages,
            feature_extractors,
            tokenizers,
            split="test",
            limit=limit,
            padding="max_length",
        )

        for idx, lang in enumerate(languages):
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processors[idx])
            eval_dataloader = DataLoader(
                processed_dataset[lang], batch_size=batch_size, collate_fn=data_collator
            )
            print(">>>>>>>>>> Language: ", lang)
            wer = run_evaluation(
                model=model, dataloader=eval_dataloader, tokenizer=tokenizers[idx]
            )
            print("WER: ", wer)


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
