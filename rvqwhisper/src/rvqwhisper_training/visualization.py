from functools import partial
from load_datasets import (
    get_cv17_asr_dataset,
    get_mls_asr_dataset,
    get_vf_asr_dataset,
    get_vp_asr_dataset,
)
from utils import load_config, DataCollatorSpeechSeq2Seq, lang2Lang, visualize_features
from rvqwhisper.model import QuantizedWhisperForConditionalGeneration
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)

import numpy as np


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


config = load_config("config.yaml")
force_compute = False
model_type = config["model_type"]
finetuned_model = config.get("finetuned_model", None)
data_root = config.get("data_root", None)
quantization_config = config["quantization"]

import torch
import os

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


from torch.utils.tensorboard import SummaryWriter

log_dir = "logs5"
writer = SummaryWriter(log_dir)
samples = []
streaming = config["evaluation"].get("streaming", False)
datasets = config["evaluation"]["datasets"]
for dataset in datasets:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dataset: ", dataset)
    if dataset == "CV17":
        func = partial(get_cv17_asr_dataset, streaming=streaming)
    if dataset == "MLS":
        func = partial(get_mls_asr_dataset, streaming=streaming)
    if dataset == "VF":
        func = partial(get_vf_asr_dataset, data_root=data_root)
    if dataset == "VPF":
        func = partial(
            get_vp_asr_dataset, streaming=streaming, cache_dir="data/voxpopuli_fr"
        )
    if dataset == "VP":
        func = partial(get_vp_asr_dataset, streaming=streaming)

    languages = config["evaluation"]["datasets"][dataset]
    tokenizers = [
        WhisperTokenizer.from_pretrained(model_type, language=lang2Lang[lang], task="transcribe")
        for lang in languages
    ]
    feature_extractors = [WhisperFeatureExtractor.from_pretrained(model_type) for _ in languages]
    processors = [
        WhisperProcessor.from_pretrained(model_type, language=lang2Lang[lang], task="transcribe")
        for lang in languages
    ]
    processed_dataset = func(
        languages, feature_extractors, tokenizers, split="test", limit=150, padding=False
    )

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torch.cuda.amp import autocast

    for idx, lang in enumerate(languages):
        print(">>>>>>>>>>>>> Language: ", lang)
        data_collator = DataCollatorSpeechSeq2Seq(processor=processors[idx])
        eval_dataloader = DataLoader(
            processed_dataset[lang], batch_size=1, collate_fn=data_collator
        )

        vis_videos = 0
        all_sims = []
        for batch in tqdm(eval_dataloader):
            # breakpoint()
            with autocast():
                encoder_outputs = model.model.model.encoder(batch["input_features"].half())
            (
                encoder_outputs_quantized,
                quantization_indices,
                quantization_loss,
                all_encoder_outputs_quantized,
            ) = model.model.model.vector_quantization(
                encoder_outputs.last_hidden_state.half(), return_all_codes=True
            )
            encoder_outputs_quantized = encoder_outputs_quantized.squeeze().detach().cpu().numpy()
            all_encoder_outputs_quantized = (
                all_encoder_outputs_quantized.squeeze(1).detach().cpu().numpy()
            )
            elm = encoder_outputs.last_hidden_state.squeeze().detach().cpu().numpy()
            if 1 and vis_videos < 2:
                # breakpoint()
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizers[idx].pad_token_id)
                decoded_labels = tokenizers[idx].batch_decode(labels, skip_special_tokens=True)
                sample_sims = [
                    cos_sim(elm[i], encoder_outputs_quantized[i]) for i in range(elm.shape[0])
                ]
                breakpoint()
                visualize_features(
                    encoder_outputs_quantized.transpose(),
                    batch["paths"][0],
                    sentence=decoded_labels[0],
                    sims=sample_sims,
                    filename=os.path.join(log_dir, f"{lang}_{dataset}_{vis_videos}_rvq.mp4"),
                )
                visualize_features(
                    elm.transpose(),
                    batch["paths"][0],
                    sentence=decoded_labels[0],
                    sims=sample_sims,
                    filename=os.path.join(log_dir, f"{lang}_{dataset}_{vis_videos}_orig.mp4"),
                )
                # writer.add_video(f'{lang}/{dataset}/{vis_videos}', sample_tensor, fps=24)

                vis_videos += 1

            vq_levels = [
                np.sum(all_encoder_outputs_quantized[: i + 1], axis=0)
                for i in range(all_encoder_outputs_quantized.shape[0])
            ]

            sims = [
                np.mean([cos_sim(elm[i], vq_levels[j][i]) for i in range(elm.shape[0])])
                for j in range(all_encoder_outputs_quantized.shape[0])
            ]
            all_sims.append(sims)
            # for ett in elm:
            sample_dict = {
                "vector": np.mean(elm, axis=0),
                # sample_dict = {'vector': ett,
                "language": lang,
                "dataset": dataset,
                "path": batch["paths"][0],
            }
            samples.append(sample_dict)

        print(
            "VQ levels cos similarities: ",
            [
                np.mean([x[j] for x in all_sims])
                for j in range(all_encoder_outputs_quantized.shape[0])
            ],
        )
        # breakpoint()

# writer.close()
# breakpoint()

embeddings = torch.from_numpy(np.asarray([x["vector"] for x in samples]))
metadata = [(x["dataset"], x["language"], x["path"]) for x in samples]
metadata_header = ["Dataset", "Language", "Path"]
writer.add_embedding(embeddings, metadata=metadata, metadata_header=metadata_header)
writer.close()
print(1)
