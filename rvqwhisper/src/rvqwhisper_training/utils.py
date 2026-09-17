from functools import partial
import numpy as np
from tqdm import tqdm
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import yaml
import os

lang2Lang = {
    "french": "French",
    "english": "English",
    "german": "German",
    "spanish": "Spanish",
    "farsi" : "Persian",
}

def prepare_text_for_whisper_labels(text):
    batch_string = text.strip()
    if batch_string and batch_string[0].isalpha():
        out = " "+batch_string[0].upper() + batch_string[1:]
    else:
        out = " "+batch_string

    return out

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

def get_mls_samples(data_dir, suffix = ".opus"):
    transcript_file = os.path.join(data_dir, 'transcripts.txt')
    
    samples = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            audio_file, transcript = line.strip().split('\t')
            audio_path = os.path.join(data_dir, "audio", audio_file.split('_')[0], audio_file.split('_')[1], audio_file+suffix)
            samples.append((audio_path, transcript))
    
    return samples

    
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # breakpoint()
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch['paths'] = [feature['path'] for feature in features]
        return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels

        return batch



    
import random
from torch.utils.data import Dataset
class CombinedDataset(Dataset):
    def __init__(self, datasets, shuffle=False):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        
        self.indices = list(range(sum(self.lengths)))
        if shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        shuffled_idx = self.indices[idx]
        for i, cumulative_length in enumerate(self.cumulative_lengths):
            if shuffled_idx < cumulative_length:
                dataset_idx = i
                sample_idx = shuffled_idx if i == 0 else shuffled_idx - self.cumulative_lengths[i-1]
                return self.datasets[dataset_idx][sample_idx]
        raise IndexError("Index out of range in CombinedDataset")
    
import threading
import time
class BufferedDataset(Dataset):
    def __init__(self, child_dataset):
        self.child_dataset = child_dataset
        self.samples = {}
        self.init_samples()

    def init_samples(self):
        print("loading samples...")
        # for idx in tqdm(range(len(self.child_dataset))):
        for idx in tqdm(range(300)):
            self.samples[idx] = self.child_dataset[idx]

        # assert len(self.samples) == len(self.child_dataset)
        print(len(self.samples))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")


# Function to pretrain codebooks using K-means
from sklearn.cluster import MiniBatchKMeans
# from sklearn.cluster import KMeans
def pretrain_codebooks(data, codebook_size):
    # data = np.reshape(data, (-1, 1280))
    if type(data) == type([]): data = np.asarray(data)
    num_samples, feature_dim = data.shape
    # Perform K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=codebook_size, max_iter=2000, batch_size=1024*8, verbose=1)
    # kmeans = KMeans(n_clusters=codebook_size, verbose=1)
    kmeans.fit(data)

    # Get centroids (codebooks)
    codebooks = kmeans.cluster_centers_
    
    return codebooks


from transformers import TrainerCallback

from torch.utils.tensorboard import SummaryWriter
class TensorBoardCallback(TrainerCallback):
    def __init__(self, config):
        self.config = config

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tb_writer = SummaryWriter(log_dir=args.logging_dir)
        self.tb_writer.add_text("config", yaml.dump(self.config), global_step=state.global_step)
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self.step_count > 0 and self.step_count % args.logging_steps == 0:
            self.tb_writer.add_scalar('train/loss_asr', kwargs["model"].model.loss_asr, state.global_step)
            for idx, loss_qt in enumerate(kwargs["model"].model.loss_qt):
               self.tb_writer.add_scalar(f'train/loss_qt{idx}', loss_qt, state.global_step)
        # if self.step_count % (100*args.logging_steps) == 0:
        #     try:
        #         self.tb_writer.add_embedding(kwargs["model"].model.vector_quantization.codebook, global_step = state.global_step, tag = "quantization/codebooks")
        #     except:
        #         self.tb_writer.add_embedding(kwargs["model"].model.model.vector_quantization.codebook, global_step = state.global_step, tag = "quantization/codebooks")

        self.step_count += 1


from torch.cuda.amp import autocast
# try:

# except: 

    # pass


class ValidationCallback(TrainerCallback):
    def __init__(self, config, valid_dataloader, processor, run_evaluation):
        self.config = config
        self.valid_dataloader = valid_dataloader
        self.processor = processor
        self.run_evaluation = run_evaluation
        self.fixed_batch = next(iter(valid_dataloader))

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tb_writer = SummaryWriter(log_dir=args.logging_dir)
        fixed_labels = self.fixed_batch["labels"].cpu().numpy()
        fixed_labels = np.where(fixed_labels != -100, fixed_labels, self.processor.tokenizer.pad_token_id)
        self.fixed_decoded_labels = self.processor.tokenizer.batch_decode(fixed_labels, skip_special_tokens=True)

        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self.step_count % self.config["finetuning"]["valid_steps"] == 0:
            # breakpoint()
            model = kwargs["model"].model
            model.eval()
            with autocast():
                with torch.no_grad():
                    generated_tokens = (
                        model.generate(
                            input_features=self.fixed_batch["input_features"].to("cuda"),
                            decoder_input_ids=self.fixed_batch["labels"][:, :4].to("cuda"),
                            max_new_tokens=255,
                        )
                        .cpu()
                        .numpy()
                    )            
            model.train()
            decoded_preds = self.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # breakpoint()
            for idx in range(len(decoded_preds)):
                self.tb_writer.add_text(f"valid_samples/{idx}", "target: " + self.fixed_decoded_labels[idx]+"\n ************************** \n"+"output: " + decoded_preds[idx].lower().strip(), global_step=state.global_step)
            # breakpoint()


        if self.step_count % (self.config["finetuning"]["eval_steps"]) == 0:
            model = kwargs["model"].model
            wer = self.run_evaluation(model=model,
                               dataloader=self.valid_dataloader, 
                               tokenizer=self.processor.tokenizer)
            self.tb_writer.add_scalar('eval/WER', wer, state.global_step)
            print("WER: ", wer)
        self.step_count += 1


from transformers import Seq2SeqTrainer, TrainingArguments
import torch

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # try:
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
        # except Exception as e:
        #     print(f"Skipping batch due to error: {e}")
        #     return None  # return zero loss for this batch
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        save only the residual vector quantization part
        """
        super().save_model(output_dir, _internal_call)
        torch.save(self.model.model.model.vector_quantization.state_dict(), os.path.join(output_dir,'rvq_model.pth'))
        # Call the parent class's save_model method to save the model
        # breakpoint()


def compute_metrics(pred, tokenizer, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def video_to_tensor(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB and then to a tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    # Convert list of frames to a PyTorch tensor
    video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, H, W, C) to (T, C, H, W)
    return video_tensor

from pydub import AudioSegment
def save_temp_wav(audio_path):
    format = audio_path.split(".")[-1]
    audio_segment = AudioSegment.from_file(audio_path, format=format)
    audio_segment.export("temp.wav", format="wav")
    

import librosa
from moviepy.editor import VideoClip, AudioFileClip
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import partial
import cv2
# Function to generate a frame with the pointer
def visualize_features(features, audio_path, sentence, sims = None, filename = "temp.mp4"):
    y, sr = librosa.load(audio_path)
    # Compute the spectrogram
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    def make_frame(t, duration, array):
        # breakpoint()
        np_std = np.asarray(array, dtype = np.float32).std()
        array_normalized = (((array - np.mean(array)) / np_std)+1)/2
        # array_normalized = np.clip(array_normalized, 0, 1)
        array_normalized = cv2.equalizeHist((array_normalized * 255).astype(np.uint8)) / 255.0
        # array_normalized = array_normalized**0.1
        fig, ax = plt.subplots()
        frame_idx = int(t/duration*array.shape[-1])
        # img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='inferno')
        img = ax.imshow(array_normalized, aspect='auto', cmap='viridis', origin='lower')
        # fig.colorbar(img, ax=ax, format="%0.001f")
        # plt.title('Spectrogram')
        # plt.title(sentence)
        # plt.ylabel('Features')
        if sims:
            img2 = ax.plot([x*array.shape[0] for x in sims], color='r')

        # Add moving pointer
        ax.axvline(x=frame_idx, color='g', linestyle='--', linewidth=2)
        
        # Convert plot to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return frame

    # Duration of the audio
    duration = librosa.get_duration(y=y, sr=sr)

    # Create the video clip with the moving pointer
    video = VideoClip(partial(make_frame, duration=duration, array=features), duration=duration)

    # Set the audio to the video clip
    format = audio_path.split(".")[-1]
    if format in ["flac"]:
        save_temp_wav(audio_path)
        audio = AudioFileClip("temp.wav")
    else:
        audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)

    # Save the video
    video.write_videofile(filename, fps=24)

    # video_to_bytes(video)
    # tensor = video_to_tensor("temp.mp4")
    # return tensor
    # return video_bytes


