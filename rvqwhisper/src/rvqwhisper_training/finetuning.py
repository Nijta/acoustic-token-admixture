import argparse
import os
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from transformers import Seq2SeqTrainingArguments
from datasets import interleave_datasets

from torch.utils.data import DataLoader
import pickle as pkl

import torch
from load_datasets import get_metadata_asr_dataset
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

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

from rvqwhisper.model import QuantizedWhisperForConditionalGeneration

from torch.cuda.amp import autocast

import evaluate

from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.audio import decode_audio

# Set the seed for random number generation
seed = 42  # You can choose any integer as a seed
torch.manual_seed(seed)
    
def main(args):
    config = load_config(args.config)
    model_type = config["model_type"]
    quantization_config = config["quantization"]

    model_name = "large-v2"
    fwm = FasterWhisperModel(model_name, device="cuda", compute_type="int8_float16")

    model = QuantizedWhisperForConditionalGeneration.from_pretrained(
        model_type,
        vector_quantization_config=quantization_config,
        load_in_8bit=True,
        device_map="auto",
    )
    model.model.faster_whisper_model = fwm
    # breakpoint()

    # model = QuantizedWhisperForConditionalGeneration.from_pretrained(model_type).to("cuda")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model = prepare_model_for_kbit_training(model)  # , output_embedding_layer_name="proj_out")
    loraconfig = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
    )

    model = get_peft_model(model, loraconfig)
    model.print_trainable_parameters()

    print(">>>>>>>>>>>> Freezing Encoder")
    for param in model.model.model.encoder.parameters():
        param.requires_grad = False
    model.print_trainable_parameters()

    datasets = config["finetuning"]["datasets"]
    limit = config["finetuning"].get("data_limit", None)
    limit_valid = config["finetuning"].get("valid_data_limit", None)

    processor = WhisperProcessor.from_pretrained(model_type, task="transcribe")

    list_processed_datasets = []
    list_processed_datasets_valid = []
    for metadata_json in datasets:
        processed_dataset, processed_dataset_valid = get_metadata_asr_dataset(
            metadata_json,
            processor,
            split=0.9,
            limit=limit,
            limit_valid=limit_valid,
            padding="max_length",
        )

        list_processed_datasets.append(processed_dataset)
        list_processed_datasets_valid.append(processed_dataset_valid)

    # breakpoint()
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    final_dataset_train = CombinedDataset(list_processed_datasets, shuffle=True)
    # final_dataloader_train = DataLoader(final_dataset_train, batch_size=config["finetuning"]["train_batch_size"], collate_fn=data_collator)

    """
    # breakpoint()
    # with open("data/processed_datasets.pkl", "wb") as f:
        # pkl.dump(final_dataset_train, f)
    # breakpoint()
    with open("data/processed_datasets.pkl", "rb") as f:
        final_dataset_train = pkl.load(f)
    """

    final_dataset_valid = CombinedDataset(list_processed_datasets_valid, shuffle=True)
    final_dataloader_valid = DataLoader(
        final_dataset_valid,
        batch_size=config["finetuning"]["eval_batch_size"],
        collate_fn=data_collator,
    )
    # final_dataset_train = interleave_datasets(list_processed_datasets)
    # final_dataset_valid = interleave_datasets(list_processed_datasets_valid)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["finetuning"]["output_dir"],  # change to a repo name of your choice
        per_device_train_batch_size=config["finetuning"]["train_batch_size"],
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=config["finetuning"]["learning_rate"],
        warmup_steps=config["finetuning"]["warmup_steps"],
        max_steps=config["finetuning"]["max_steps"],
        gradient_checkpointing=True,
        max_grad_norm=config["finetuning"].get("max_grad_norm", 1.0),
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=config["finetuning"]["eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=config["finetuning"]["max_length"],
        save_steps=config["finetuning"]["save_steps"],
        eval_steps=config["finetuning"]["eval_steps"],
        logging_steps=config["finetuning"]["logging_steps"],
        report_to=["tensorboard"],
        greater_is_better=False,
    )

    tensorboard_callback = TensorBoardCallback(config=config)
    from evaluation import run_evaluation

    validation_callback = ValidationCallback(
        config=config,
        valid_dataloader=final_dataloader_valid,
        processor=processor,
        run_evaluation=run_evaluation,
    )
    trainer = CustomSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=final_dataset_train,
        eval_dataset=final_dataset_valid,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[tensorboard_callback, validation_callback],
    )

    resume_from_checkpoint = config["finetuning"].get("resume_from_checkpoint", False)
    if resume_from_checkpoint == "True":
        resume_from_checkpoint = True

    if not resume_from_checkpoint:
        model.config.use_cache = False

    with autocast():
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(config["finetuning"]["output_dir"], "peft_model.pth"),
    )


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
