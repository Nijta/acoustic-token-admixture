import argparse
import os
import tarfile
import requests
from tqdm import tqdm

from functools import partial

from utils import gen_from_iterable_dataset, prepare_text_for_whisper_labels

import soundfile as sf
import numpy as np
from datasets import load_dataset, DatasetDict, Audio

lang2url = {
    "english": "https://dl.fbaipublicfiles.com/mls/mls_english_opus.tar.gz",
    "french": "https://dl.fbaipublicfiles.com/mls/mls_french_opus.tar.gz",
    "german": "https://dl.fbaipublicfiles.com/mls/mls_german_opus.tar.gz"
}
def download_file(url, dest):
    print(f"Downloading {url} to {dest}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(dest, 'wb') as file:
        for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB'):
            file.write(data)

def extract_tar_gz(file_path, extract_path):
    print(f"Extracting {file_path} to {extract_path}")
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

from datasets import DatasetDict, IterableDataset, Dataset


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch




def get_local_mls_asr_dataset(languages, feature_extractors, tokenizers, data_root, split = "train"):

    dataset_dict = {}
    for idx, lang in enumerate(languages):
        data_dir = os.path.join(data_root, lang, lang2url[lang].split("/")[-1].split(".")[0], split)
        samples = get_mls_samples(data_dir)
        iterable_dataset = IterableDataset.from_generator(sample_generator(samples, feature_extractors[idx], tokenizers[idx]))
        dataset_dict[lang] = iterable_dataset
    dataset_dict = DatasetDict(dataset_dict)

    return dataset_dict

def get_vf_asr_dataset(languages, feature_extractors, tokenizers, data_root, limit = None, split = None, padding = True):

    dataset_dict = {}
    for idx, lang in enumerate(languages):
        if lang == "french":
            samples = get_vf_samples(data_root, "VF_french_mos33.json", split = split)
            #breakpoint()
            if limit:
                samples = samples[:limit]
            
            iterable_dataset = IterableDataset.from_generator(sample_generator(samples, feature_extractors[idx], tokenizers[idx], padding = padding))
            ds = Dataset.from_generator(partial(gen_from_iterable_dataset, iterable_dataset))
            # examples = [example for example in iterable_dataset]
            # data_dict = {key: [example[key] for example in examples] for key in examples[0]}
            # dataset = Dataset.from_dict(data_dict)
            dataset_dict[lang] = ds


    dataset_dict = DatasetDict(dataset_dict)
    
    return dataset_dict


cv17_lang2lang = {'english': 'en',
                  'french': 'fr',
                  'german': 'de',
                  'spanish': 'es'}
def get_cv17_asr_dataset(languages, feature_extractors, tokenizers, split = "train", limit = None, cache_dir = None, streaming = False, padding = True):
    common_voice = DatasetDict()

    for lang in languages:
        common_voice[lang] = load_dataset("mozilla-foundation/common_voice_17_0", cv17_lang2lang[lang], split=split, use_auth_token=True,cache_dir=cache_dir, streaming=streaming)
        if streaming:
            samples = []
            for i, sample in tqdm(enumerate(common_voice[lang])):
                if limit and i >= limit:
                    break
                samples.append(sample)

            # Convert the collected samples to a Dataset object
            common_voice[lang] = Dataset.from_dict({key: [sample[key] for sample in samples] for key in samples[0]})
        elif limit:
            common_voice[lang] = common_voice[lang].select(range(limit))

    common_voice = common_voice.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
    )
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
    def get_prepare_func(ffeature_extractor, ttokenizer, padding = True):
        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]
            batch['path'] = (batch['audio']['path'])

            # compute log-Mel input features from input audio array
            batch["input_features"] = ffeature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], padding=padding).input_features[0]

            # encode target text to label ids
            batch["labels"] = ttokenizer(prepare_text_for_whisper_labels(batch["sentence"])).input_ids
            return batch
        return prepare_dataset
    for idx, lang in enumerate(languages):
        common_voice[lang] = common_voice[lang].map(get_prepare_func(feature_extractors[idx], tokenizers[idx], padding=padding), remove_columns=common_voice.column_names[lang], num_proc=1)
    return common_voice

whisper_lang2lang = {'english': 'en',
                     'french': 'fr',
                     'german': 'de',
                     'spanish': 'es'}

def sample_generator(samples, processor, target_sample_rate=16000, padding=True):
    def generator():
        for sample in samples:
            # try:
            processor.tokenizer.set_prefix_tokens(whisper_lang2lang[sample.get("language", "english")])
            # print(sample)

            # Load the audio file directly at the target sample rate
            waveform, sample_rate = sf.read(sample["path"])

            # If the original sample rate is needed for feature extraction
            if sample_rate != target_sample_rate:
                num_samples = int(len(waveform) * target_sample_rate / sample_rate)
                waveform = np.interp(np.linspace(0, len(waveform), num_samples), 
                                        np.arange(len(waveform)), 
                                        waveform)
                sample_rate = target_sample_rate
            
            # Extract features
            input_features = processor.feature_extractor(waveform, sampling_rate=sample_rate, padding=padding, return_tensors="pt").input_features[0]
            
            labels = processor.tokenizer(sample["sentence"], return_tensors="pt", padding=False).input_ids.squeeze(0)

            yield {
                "input_features": input_features,
                "labels": labels,
                "path": sample["path"]
            }
            # except Exception as e:
            #     print(f"Error processing sample {sample['path']}: {e}")
            #     continue
            
    return generator


import json
def get_metadata_asr_dataset(metadata_json, processor, split = 0.9, limit = None, limit_valid = None, padding = True):
    with open(metadata_json, "r") as f:
        samples = json.load(f)
    if split:
        samples_train = samples[:int(len(samples)*split)]
        samples_valid = samples[int(len(samples)*split):]
        if limit:
            samples_train = samples_train[:limit]
        if limit_valid:
            samples_valid = samples_valid[:limit_valid]

        # breakpoint()
        generator_train = sample_generator(samples_train, processor, padding = padding)
        generator_valid = sample_generator(samples_valid, processor, padding = padding)
        ds_train = Dataset.from_generator(generator_train)
        ds_valid = Dataset.from_generator(generator_valid)

        return ds_train, ds_valid

    else:
        if limit:
            samples = samples[:limit]
        iterable_dataset = IterableDataset.from_generator(sample_generator(samples, processor, padding = padding))
        ds = Dataset.from_generator(partial(gen_from_iterable_dataset, iterable_dataset))
        return samples

mls_lang2lang = {'french': 'french', 'german': 'german', 'english': 'english','spanish':'spanish'}
def get_mls_asr_dataset(languages, feature_extractors, tokenizers, split = "train", limit = None, cache_dir = None, streaming = False, padding = True):
    mls = DatasetDict()
    for lang in languages:
        mls[lang] = load_dataset("facebook/multilingual_librispeech", mls_lang2lang[lang], split=split, use_auth_token=True, cache_dir=cache_dir, streaming = streaming)
        if streaming:
            samples = []
            for i, sample in tqdm(enumerate(mls[lang])):
                if limit and i >= limit:
                    break
                samples.append(sample)

            # Convert the collected samples to a Dataset object
            mls[lang] = Dataset.from_dict({key: [sample[key] for sample in samples] for key in samples[0]})
        elif limit:
            mls[lang] = mls[lang].select(range(limit))
    
    mls = mls.remove_columns(
        ["file", "speaker_id", "chapter_id", "id"]
    )
    mls = mls.cast_column("audio", Audio(sampling_rate=16000))
    
    def get_prepare_func(ffeature_extractor, ttokenizer, padding = True):
        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]
            batch['path'] = (batch['audio']['path'])
            # compute log-Mel input features from input audio array
            batch["input_features"] = ffeature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], padding=padding).input_features[0]

            # encode target text to label ids
            batch["labels"] = ttokenizer(prepare_text_for_whisper_labels(batch["text"])).input_ids

            return batch
        return prepare_dataset
    
    for idx, lang in enumerate(languages):
        mls[lang] = mls[lang].map(get_prepare_func(feature_extractors[idx], tokenizers[idx], padding=padding), remove_columns=mls.column_names[lang], num_proc=1)
    return mls


'''
def main(args):
    config = load_config(args.config)
    force_compute = args.force_compute

    data_root = config['datasets']['data_root']
    languages = config['datasets']['languages']

    for lang in languages:
        file_name = os.path.basename(lang2url[lang])
        file_path = os.path.join(data_root, file_name)
        extract_dir = os.path.join(data_root, lang)
        if (not os.path.exists(file_path) and not os.path.exists(extract_dir)) or force_compute:
            download_file(lang2url[lang], file_path)
        if not os.path.exists(extract_dir) or force_compute:
            os.makedirs(extract_dir, exist_ok=True)
            extract_tar_gz(file_path, extract_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset based on configurations.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--force_compute', type=bool, default=False, help="recompute all steps even if already computed")

    args = parser.parse_args()
    main(args)

'''