import setuptools
import re
import os
import json
import torch
import torch.nn as nn
import numpy as np
np.set_printoptions(threshold=np.inf)
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging
# import wandb  # Optional for logging
try:
    from aligner.src.articulatory_features import generate_feature_table, get_phone_to_id, get_feature_to_index_lookup
except:
    from articulatory_features import generate_feature_table, get_phone_to_id, get_feature_to_index_lookup


from transphone.g2p import read_g2p

class Aligner:
    def __init__(self, lang, whisper_wrapper=None):
        self.whisper = whisper_wrapper
        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_phone_to_id()
        self.phone_to_id[" "] = 0
        self.phone_to_id["~"] = 1
        self.phone_to_id["?"] = 2
        self.phone_to_id["!"] = 3
        self.phone_to_id["."] = 4
        self.phone_to_id["#"] = 5
        self.transphone = read_g2p(device="cuda")
        self.g2p_lang = lang
        self.rising_perms = list()
        self.falling_perms = list()
        self.peaking_perms = list()
        self.dipping_perms = list()

        self.expand_abbreviations = lambda x: x
        register_to_height = {
            "˥": 5,
            "˦": 4,
            "˧": 3,
            "˨": 2,
            "˩": 1
        }
        for first_tone in ["˥", "˦", "˧", "˨", "˩"]:
            for second_tone in ["˥", "˦", "˧", "˨", "˩"]:
                if register_to_height[first_tone] > register_to_height[second_tone]:
                    self.falling_perms.append(first_tone + second_tone)
                else:
                    self.rising_perms.append(first_tone + second_tone)
                for third_tone in ["˥", "˦", "˧", "˨", "˩"]:
                    if register_to_height[first_tone] > register_to_height[second_tone] < register_to_height[third_tone]:
                        self.dipping_perms.append(first_tone + second_tone + third_tone)
                    elif register_to_height[first_tone] < register_to_height[second_tone] > register_to_height[third_tone]:
                        self.peaking_perms.append(first_tone + second_tone + third_tone)
    
    def extract_features(self, audio_path):
        bn = self.whisper.compute_bottleneck(audio_path, vectorize=True)
        return bn
    
    def get_phone_string(self, text, include_eos_symbol=True, for_feature_extraction=False, for_plot_labels=False, split_words=False):
        if text == "":
            return ""
        # expand abbreviations
        utt = self.expand_abbreviations(text)

        replacements = [
            # punctuation in languages with non-latin script
            ("。", "~"),
            ("，", "~"),
            ("【", '~'),
            ("】", '~'),
            ("、", "~"),
            ("‥", "~"),
            ("؟", "~"),
            ("،", "~"),
            ("“", '~'),
            ("”", '~'),
            ("؛", "~"),
            ("《", '~'),
            ("》", '~'),
            ("？", "~"),
            ("！", "~"),
            (" ：", "~"),
            (" ；", "~"),
            ("－", "~"),
            ("·", " "),
            ("`", ""),
            # symbols that indicate a pause or silence
            ('"', "~"),
            (" - ", "~ "),
            ("- ", "~ "),
            ("-", ""),
            ("…", "~"),
            (":", "~"),
            (";", "~"),
            (",", "~")  # make sure this remains the final one when adding new ones
        ]
        for replacement in replacements:
            utt = utt.replace(replacement[0], replacement[1])
        utt = re.sub("~+", "~", utt)
        utt = re.sub(r"\s+", " ", utt)
        utt = re.sub(r"\.+", ".", utt)
        chunk_list = list()
        for chunk in utt.split("~"):
            # unfortunately the transphone tokenizer is not suited for any languages besides English it seems
            # this is not much better, but maybe a little.
            word_list = list()
            for word_by_whitespace in chunk.split():
                word_list.append(self.transphone.inference(word_by_whitespace, self.g2p_lang))
            chunk_list.append(" ".join(["".join(word) for word in word_list]))
        phones = "~ ".join(chunk_list)
        
        # Unfortunately tonal languages don't agree on the tone, most tonal
        # languages use different tones denoted by different numbering
        # systems. At this point in the script, it is attempted to unify
        # them all to the tones in the IPA standard.
        
        # more of this handling for more tonal languages can be added here, simply make an elif statement and check for the language.
        return self.postprocess_phoneme_string(phones, for_feature_extraction, include_eos_symbol, for_plot_labels)
    
    def postprocess_phoneme_string(self, phoneme_string, for_feature_extraction, include_eos_symbol, for_plot_labels):
        """
        Takes as input a phoneme string and processes it to work best with the way we represent phonemes as featurevectors
        """
        replacements = [
            # punctuation in languages with non-latin script
            ("。", "."),
            ("，", ","),
            ("【", '"'),
            ("】", '"'),
            ("、", ","),
            ("‥", "…"),
            ("؟", "?"),
            ("،", ","),
            ("“", '"'),
            ("”", '"'),
            ("؛", ","),
            ("《", '"'),
            ("》", '"'),
            ("？", "?"),
            ("！", "!"),
            (" ：", ":"),
            (" ；", ";"),
            ("－", "-"),
            ("·", " "),
            # latin script punctuation
            ("/", " "),
            ("—", ""),
            ("(", "~"),
            (")", "~"),
            ("...", "…"),
            ("\n", ", "),
            ("\t", " "),
            ("¡", ""),
            ("¿", ""),
            ("«", '"'),
            ("»", '"'),
            # unifying some phoneme representations
            ("N", "ŋ"),  # somehow transphone doesn't transform this to IPA
            ("ɫ", "l"),  # alveolopalatal
            ("ɚ", "ə"),
            ("g", "ɡ"),
            ("ε", "e"),
            ("ʦ", "ts"),
            ("ˤ", "ˁ"),
            ('ᵻ', 'ɨ'),
            ("ɧ", "ç"),  # velopalatal
            ("ɥ", "j"),  # labiopalatal
            ("ɬ", "s"),  # lateral
            ("ɮ", "z"),  # lateral
            ('ɺ', 'ɾ'),  # lateral
            ('ʲ', 'j'),  # decomposed palatalization
            ('\u02CC', ""),  # secondary stress
            ('\u030B', "˥"),
            ('\u0301', "˦"),
            ('\u0304', "˧"),
            ('\u0300', "˨"),
            ('\u030F', "˩"),
            ('\u0302', "⭨"),
            ('\u030C', "⭧"),
            ("꜖", "˩"),
            ("꜕", "˨"),
            ("꜔", "˧"),
            ("꜓", "˦"),
            ("꜒", "˥"),
            # symbols that indicate a pause or silence
            ('"', "~"),
            (" - ", "~ "),
            ("- ", "~ "),
            ("-", ""),
            ("…", "."),
            (":", "~"),
            (";", "~"),
            (",", "~")  # make sure this remains the final one when adding new ones
        ]
        unsupported_ipa_characters = {'̙', '̯', '̤', '̩', '̠', '̟', 'ꜜ', '̽', '|', '•', '↘',
                                      '‖', '‿', 'ᷝ', 'ᷠ', '̚', '↗', 'ꜛ', '̻', '̘', '͡', '̺'}
        #  https://en.wikipedia.org/wiki/IPA_number
        for char in unsupported_ipa_characters:
            replacements.append((char, ""))

        if not for_feature_extraction:
            # in case we want to plot etc., we only need the segmental units, so we remove everything else.
            replacements = replacements + [
                ('\u02C8', ""),  # primary stress
                ('\u02D0', ""),  # lengthened
                ('\u02D1', ""),  # half-length
                ('\u0306', ""),  # shortened
                ("˥", ""),  # very high tone
                ("˦", ""),  # high tone
                ("˧", ""),  # mid tone
                ("˨", ""),  # low tone
                ("˩", ""),  # very low tone
                ('\u030C', ""),  # rising tone
                ('\u0302', ""),  # falling tone
                ('⭧', ""),  # rising
                ('⭨', ""),  # falling
                ('⮃', ""),  # dipping
                ('⮁', ""),  # peaking
                ('̃', ""),  # nasalizing
                ("̧", ""),  # palatalized
                ("ʷ", ""),  # labialized
                ("ʰ", ""),  # aspirated
                ("ˠ", ""),  # velarized
                ("ˁ", ""),  # pharyngealized
                ("ˀ", ""),  # glottalized
                ("ʼ", ""),  # ejective
                ("̹", ""),  # rounding
                ("̞", ""),  # open
                ("̪", ""),  # dental
                ("̬", ""),  # voiced
                ("̝", ""),  # closed
                ("̰", ""),  # laryngalization
                ("̈", ""),  # centralization
                ("̜", ""),  # unrounded
                ("̥", ""),  # voiceless
            ]
        for replacement in replacements:
            phoneme_string = phoneme_string.replace(replacement[0], replacement[1])
        phones = re.sub("~+", "~", phoneme_string)
        phones = re.sub(r"\s+", " ", phones)
        phones = re.sub(r"\.+", ".", phones)
        phones = phones.lstrip("~").rstrip("~")

        # peaking tones
        for peaking_perm in self.peaking_perms:
            phones = phones.replace(peaking_perm, "⮁".join(peaking_perm))
        # dipping tones
        for dipping_perm in self.dipping_perms:
            phones = phones.replace(dipping_perm, "⮃".join(dipping_perm))
        # rising tones
        for rising_perm in self.rising_perms:
            phones = phones.replace(rising_perm, "⭧".join(rising_perm))
        # falling tones
        for falling_perm in self.falling_perms:
            phones = phones.replace(falling_perm, "⭨".join(falling_perm))

        phones += "~"  # adding a silence in the end during inference produces more natural sounding prosody
        phones += "#"
        # if not self.use_word_boundaries:
            # phones = phones.replace(" ", "")

        phones = "~" + phones
        phones = re.sub("~+", "~", phones)

        return phones
    
    def string_to_tensor(self, text, view=False, device="cpu", handle_missing=True, input_phonemes=False, return_phonemes=False, split_words = False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        if input_phonemes:
            phones = text
        else:
            phones = self.get_phone_string(text=text, include_eos_symbol=True, for_feature_extraction=True, split_words=split_words)
        phones = phones.replace("ɚ", "ə").replace("ᵻ", "ɨ")
        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        stressed_flag = False
        goodchars = ''
        for char in phones:
            # affects following phoneme -----------------
            if char.strip() == '\u02C8':
                # primary stress
                stressed_flag = True
            # affects previous phoneme -----------------
            elif char.strip() == '\u02D0':
                # lengthened
                phones_vector[-1][get_feature_to_index_lookup()["lengthened"]] = 1
            elif char.strip() == '\u02D1':
                # half length
                phones_vector[-1][get_feature_to_index_lookup()["half-length"]] = 1
            elif char.strip() == '\u0306':
                # shortened
                phones_vector[-1][get_feature_to_index_lookup()["shortened"]] = 1
            elif char.strip() == '̃' and phones_vector[-1][get_feature_to_index_lookup()["nasal"]] != 1:
                # nasalized (vowel)
                phones_vector[-1][get_feature_to_index_lookup()["nasal"]] = 2
            elif char.strip() == "̧" != phones_vector[-1][get_feature_to_index_lookup()["palatal"]] != 1:
                # palatalized
                phones_vector[-1][get_feature_to_index_lookup()["palatal"]] = 2
            elif char.strip() == "ʷ" and phones_vector[-1][get_feature_to_index_lookup()["labial-velar"]] != 1:
                # labialized
                phones_vector[-1][get_feature_to_index_lookup()["labial-velar"]] = 2
            elif char.strip() == "ʰ" and phones_vector[-1][get_feature_to_index_lookup()["aspirated"]] != 1:
                # aspirated
                phones_vector[-1][get_feature_to_index_lookup()["aspirated"]] = 2
            elif char.strip() == "ˠ" and phones_vector[-1][get_feature_to_index_lookup()["velar"]] != 1:
                # velarized
                phones_vector[-1][get_feature_to_index_lookup()["velar"]] = 2
            elif char.strip() == "ˁ" and phones_vector[-1][get_feature_to_index_lookup()["pharyngal"]] != 1:
                # pharyngealized
                phones_vector[-1][get_feature_to_index_lookup()["pharyngal"]] = 2
            elif char.strip() == "ˀ" and phones_vector[-1][get_feature_to_index_lookup()["glottal"]] != 1:
                # glottalized
                phones_vector[-1][get_feature_to_index_lookup()["glottal"]] = 2
            elif char.strip() == "ʼ" and phones_vector[-1][get_feature_to_index_lookup()["ejective"]] != 1:
                # ejective
                phones_vector[-1][get_feature_to_index_lookup()["ejective"]] = 2
            elif char.strip() == "̹" and phones_vector[-1][get_feature_to_index_lookup()["rounded"]] != 1:
                # rounding
                phones_vector[-1][get_feature_to_index_lookup()["rounded"]] = 2
            elif char.strip() == "̞" and phones_vector[-1][get_feature_to_index_lookup()["open"]] != 1:
                # open
                phones_vector[-1][get_feature_to_index_lookup()["open"]] = 2
            elif char.strip() == "̪" and phones_vector[-1][get_feature_to_index_lookup()["dental"]] != 1:
                # dental
                phones_vector[-1][get_feature_to_index_lookup()["dental"]] = 2
            elif char.strip() == "̬" and phones_vector[-1][get_feature_to_index_lookup()["voiced"]] != 1:
                # voiced
                phones_vector[-1][get_feature_to_index_lookup()["voiced"]] = 2
            elif char.strip() == "̝" and phones_vector[-1][get_feature_to_index_lookup()["close"]] != 1:
                # closed
                phones_vector[-1][get_feature_to_index_lookup()["close"]] = 2
            elif char.strip() == "̰" and phones_vector[-1][get_feature_to_index_lookup()["glottal"]] != 1 and phones_vector[-1][get_feature_to_index_lookup()["epiglottal"]] != 1:
                # laryngalization
                phones_vector[-1][get_feature_to_index_lookup()["glottal"]] = 2
                phones_vector[-1][get_feature_to_index_lookup()["epiglottal"]] = 2
            elif char.strip() == "̈" and phones_vector[-1][get_feature_to_index_lookup()["central"]] != 1:
                # centralization
                phones_vector[-1][get_feature_to_index_lookup()["central"]] = 2
            elif char.strip() == "̜" and phones_vector[-1][get_feature_to_index_lookup()["unrounded"]] != 1:
                # unrounded
                phones_vector[-1][get_feature_to_index_lookup()["unrounded"]] = 2
            elif char.strip() == "̥" and phones_vector[-1][get_feature_to_index_lookup()["unvoiced"]] != 1:
                # voiceless
                phones_vector[-1][get_feature_to_index_lookup()["unvoiced"]] = 2
            elif char.strip() == "˥":
                # very high tone
                phones_vector[-1][get_feature_to_index_lookup()["very-high-tone"]] = 1
            elif char.strip() == "˦":
                # high tone
                phones_vector[-1][get_feature_to_index_lookup()["high-tone"]] = 1
            elif char.strip() == "˧":
                # mid tone
                phones_vector[-1][get_feature_to_index_lookup()["mid-tone"]] = 1
            elif char.strip() == "˨":
                # low tone
                phones_vector[-1][get_feature_to_index_lookup()["low-tone"]] = 1
            elif char.strip() == "˩":
                # very low tone
                phones_vector[-1][get_feature_to_index_lookup()["very-low-tone"]] = 1
            elif char.strip() == "⭧":
                # rising tone
                phones_vector[-1][get_feature_to_index_lookup()["rising-tone"]] = 1
            elif char.strip() == "⭨":
                # falling tone
                phones_vector[-1][get_feature_to_index_lookup()["falling-tone"]] = 1
            elif char.strip() == "⮁":
                # peaking tone
                phones_vector[-1][get_feature_to_index_lookup()["peaking-tone"]] = 1
            elif char.strip() == "⮃":
                # dipping tone
                phones_vector[-1][get_feature_to_index_lookup()["dipping-tone"]] = 1
            else:
                if handle_missing:
                    try:
                        phones_vector.append(self.phone_to_vector[char].copy())
                        goodchars += char
                    except KeyError:
                        print("unknown phoneme: {}".format(char))
                else:
                    phones_vector.append(self.phone_to_vector[char].copy())  # leave error handling to elsewhere
                # the following lines try to emulate whispering by removing all voiced features
                # phones_vector[-1][get_feature_to_index_lookup()["voiced"]] = 0
                # phones_vector[-1][get_feature_to_index_lookup()["unvoiced"]] = 1
                # the following lines explore what would happen, if the system is told to produce sounds a human cannot
                # for dim, _ in enumerate(phones_vector[-1]):
                #     phones_vector[-1][dim] = 1
                if stressed_flag:
                    stressed_flag = False
                    phones_vector[-1][get_feature_to_index_lookup()["stressed"]] = 1
        if return_phonemes:
            return goodchars
        return torch.Tensor(phones_vector, device=device)
    
    def extract_phonemes(self, transcription):
        ps = self.string_to_tensor(transcription, return_phonemes=True)
        ids = [self.phone_to_id[x] for x in ps[1:-2]]
        return [0]+ids+[0]
    

# Configuration
try:
    from config import Config 
except:
    from aligner.src.config import Config

# Dataset Handling
class PhonemeDataset(Dataset):
    def __init__(self, samples, aligner, features_path=None):
        self.samples = samples
        self.aligner = aligner
        self.features_path = features_path
        
    def __len__(self):
        return len(self.samples)
    def f_read_raw_mat(self, filename, col, data_format="f4", end="l"):
        f = open(filename, "rb")
        if end == "l":
            data_format = "<" + data_format
        datatype = np.dtype((data_format, (col,)))
        data = np.fromfile(f, dtype=datatype)
        f.close()
        if data.ndim == 2 and data.shape[1] == 1:
            return data[:, 0]
        else:
            return data
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract features
        audio_path = sample['path']
        if self.features_path is None:
            features = self.aligner.extract_features(audio_path)
        else:
            tokens_filename = os.path.join(self.features_path, sample['prefix']+os.path.basename(audio_path).split(".")[0] + ".ppg")
            tokens = self.f_read_raw_mat(tokens_filename, 8)  
            features = self.aligner.whisper.decode_from_codebook_indices(tokens[None,...])[0]
            
        # Extract phonemes
        transcription = sample['sentence']
        phonemes = self.aligner.extract_phonemes(transcription)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'phonemes': torch.tensor(phonemes, dtype=torch.long)
        }

def collate_fn(batch):
    # Sort batch by feature length (descending) for packed sequences
    batch.sort(key=lambda x: len(x['features']), reverse=True)
    
    features = [item['features'] for item in batch]
    phonemes = [item['phonemes'] for item in batch]
    
    features_lens = torch.tensor([len(f) for f in features], dtype=torch.long)
    phonemes_lens = torch.tensor([len(p) for p in phonemes], dtype=torch.long)
    
    features_pad = nn.utils.rnn.pad_sequence(features, batch_first=True)
    phonemes_pad = nn.utils.rnn.pad_sequence(phonemes, batch_first=True)
    
    return {
        'features': features_pad.to(device=Config.DEVICE, non_blocking=True),
        'features_lens': features_lens.to(Config.DEVICE, non_blocking=True),
        'phonemes': phonemes_pad.to(Config.DEVICE, non_blocking=True),
        'phonemes_lens': phonemes_lens.to(Config.DEVICE, non_blocking=True)
    }

# Model Architecture
class CTCHead(nn.Module):
    def __init__(self, input_dim, num_phonemes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.linear = nn.Linear(512, num_phonemes + 1)
        
    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # (T, B, D)
        return nn.functional.log_softmax(self.linear(x), dim=-1)

# Training Infrastructure
class CTCTrainer:
    def __init__(self, config, aligner):
        self.config = config
        self.aligner = aligner
        self.model = CTCHead(config.INPUT_DIM, config.NUM_PHONEMES).to(config.DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.criterion = nn.CTCLoss(blank=Config.NUM_PHONEMES)
        
        # Setup datasets
        self.train_loader, self.val_loader = self.prepare_datasets()
        
        # Logging
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.scaler = torch.cuda.amp.GradScaler()

    def prepare_datasets(self):
        # Load all datasets
        all_samples = []
        for path, pref in zip(self.config.DATASET_PATHS, self.config.PREFS):
            with open(path) as f:
                all_samples += [{**d, 'prefix': pref} for d in json.load(f)]
                
        # Split datasets
        train_samples, val_samples = train_test_split(
            all_samples, 
            train_size=self.config.TRAIN_RATIO,
            random_state=self.config.RANDOM_SEED
        )
        
        train_dataset = PhonemeDataset(train_samples, self.aligner, features_path=config.FEATURES_PATH)
        val_dataset = PhonemeDataset(val_samples[:256], self.aligner, features_path=config.FEATURES_PATH)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            # num_workers=16, 
            # pin_memory=True,
            # persistent_workers=True,
            # prefetch_factor=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            collate_fn=collate_fn,
            # num_workers=16, 
            # pin_memory=True,
            # persistent_workers=True,
            # prefetch_factor=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            
            log_probs = self.model(batch['features'])
            loss = self.criterion(
                log_probs,
                batch['phonemes'],
                batch['features_lens'],
                batch['phonemes_lens']
            )
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # More efficient
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                log_probs = self.model(batch['features'])
                loss = self.criterion(
                    log_probs,
                    batch['phonemes'],
                    batch['features_lens'],
                    batch['phonemes_lens']
                )
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                
        return total_loss / len(self.val_loader)
    
    def train(self):
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                
            # Early stopping
            if self.early_stop_counter >= self.config.PATIENCE:
                print("Early stopping triggered")
                break
            
            # Update plots
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.plot_losses(train_losses, val_losses)
            
    def save_model(self):
        path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self):
        path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pth")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
        
    def plot_losses(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.CHECKPOINT_DIR, 'loss_curve.png'))
        plt.close()

# Inference Code
class PhonemeInference:
    def __init__(self, aligner, model_path):
        self.aligner = aligner
        self.model = CTCHead(Config.INPUT_DIM, Config.NUM_PHONEMES)
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.eval()
        self.xos_tensor = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unsqueeze(0)
        
        self.id_to_phone = {v: k for k, v in self.aligner.phone_to_id.items()}
        self.phone_to_artics = generate_feature_table()
        
    def predict(self, bottleneck):
        # Extract features
        # features = self.aligner.extract_features(audio_path)
        bottleneck = torch.tensor(bottleneck, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            log_probs = self.model(bottleneck)
            predictions = torch.argmax(log_probs, dim=-1)
            
        # Decode CTC output
        decoded = []
        prev = 0
        for idx in predictions.squeeze().tolist():
            if idx != prev and idx != Config.NUM_PHONEMES:  # Config.NUM_PHONEMES is blank
                decoded.append(self.id_to_phone[idx])
                prev = idx
            else:
                decoded.append(self.id_to_phone[prev])
        
        artics = np.concatenate([np.asarray(self.phone_to_artics[x])[None, ...] if x != " " else self.xos_tensor
                                 for x in decoded], axis=0)
        return torch.from_numpy(artics).float(), decoded

class ForcedAligner:
    def __init__(self, model, aligner, blank=0):
        self.model = model.eval()
        self.aligner = aligner
        self.blank = blank
        self.phone_to_id = aligner.phone_to_id
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
        self.phone_to_artics = generate_feature_table()
        self.xos_tensor = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unsqueeze(0)

    def align(self, audio_path, target_ids):
        """Returns aligned (phoneme, start_time, end_time) tuples"""
        if type(audio_path) != type(np.zeros(0)):
            features = self.aligner.extract_features(audio_path)
            if type(features) == type([]):
                features = features[0]
        else:
            features = audio_path
            if features.shape[-1] == 8:
                features = self.aligner.whisper.decode_from_codebook_indices(features[None,...])[0]

        # breakpoint()
        features_t = torch.tensor(features).unsqueeze(0).to(Config.DEVICE)
        
        target = [self.blank] + [x for p in target_ids for x in (p, self.blank)]
        with torch.no_grad():
            log_probs = self.model(features_t).squeeze(1)  # (T, C)
        
        alignment = self._forced_align(log_probs, target)
        if all([x==111 for x in alignment]):
            alignment = self._forced_align2(log_probs, target)
                
        result = self._compute_alignment(alignment)
        artics = np.concatenate([np.asarray(self.phone_to_artics[x[0]])[None, ...] if x[0] != " " else self.xos_tensor
                                 for x in result], axis=0)
        ali = np.asarray([x[2] - x[1] for x in result])
        return torch.from_numpy(ali), torch.from_numpy(artics)
    
    def _forced_align(self, log_probs, target):
        T, C = log_probs.shape
        S = len(target)
        device = log_probs.device

        # Precompute all emissions for each target token (T, S)
        emissions = log_probs[:, target]  # shape: (T, S)

        # Initialize DP: at time 0, only state 0 is allowed.
        dp_prev = torch.full((S,), -float('inf'), device=device)
        dp_prev[0] = emissions[0, 0]
        
        # Backpointers: to record which previous state was chosen
        backpointers = torch.zeros((T, S), dtype=torch.long, device=device)

        for t in range(1, T):
            # For each state, compute two candidates:
            #   - Staying at the same target token: dp_prev (score remains at s)
            #   - Transitioning from the previous token: shift dp_prev by one (with -inf prepended)
            dp_stay = dp_prev
            dp_shift = torch.cat([torch.full((1,), -float('inf'), device=device), dp_prev[:-1]])
            
            # For each state s, choose the option with the higher score.
            max_prev = torch.max(dp_stay, dp_shift)
            # If dp_shift is higher, then we came from s-1.
            max_mask = (dp_shift > dp_stay).long()
            backpointers[t] = torch.arange(S, device=device) - max_mask
            # Update dp for time t by adding the emission for each state.
            dp_prev = max_prev + emissions[t]

        # Instead of choosing the best state via argmax, force the final state to be the last token.
        s = S - 1
        path = []
        # Backtrack from the last time step to time 0.
        for t in reversed(range(T)):
            path.append(s)
            s = backpointers[t, s].item()
        # Reverse the path so that it is in chronological order.
        aligned_targets = [target[s] for s in reversed(path)]
        return aligned_targets
    
    def _forced_align2(self, log_probs, target):
        T, C = log_probs.shape
        S = len(target)
        device = log_probs.device

        # Precompute emissions: (T, S)
        emissions = log_probs[:, target]  # assume target contains indices

        # Initialize DP: at time 0, only state 0 is allowed.
        dp_prev = torch.full((S,), -float('inf'), device=device)
        dp_prev[0] = emissions[0, 0]
        
        # Backpointers: record the previous state chosen
        backpointers = torch.zeros((T, S), dtype=torch.long, device=device)

        for t in range(1, T):
            # Compute the two candidate scores:
            dp_stay = dp_prev  # staying in the same token
            dp_shift = torch.cat([torch.full((1,), -float('inf'), device=device), dp_prev[:-1]])
            
            s_indices = torch.arange(S, device=device)
            t_remaining = T - t  # frames left including current t
            # For each state, if the remaining frames exactly equal the remaining tokens,
            # we must transition (except at state 0).
            need_transition = (t_remaining == (S - s_indices)) & (s_indices > 0)
            
            # Normally, choose the max over staying and shifting.
            normal_choice = torch.max(dp_stay, dp_shift)
            # Where forced transition is needed, use dp_shift.
            candidate = torch.where(need_transition, dp_shift, normal_choice)
            
            # Backpointer: if we took shift then record s-1, else s.
            transition_choice = (dp_shift > dp_stay).long()
            bp = s_indices - transition_choice
            # Override with forced transition where needed.
            bp = torch.where(need_transition, s_indices - 1, bp)
            backpointers[t] = bp

            # Update dp
            dp_prev = candidate + emissions[t]

        # Force final state to be the last token
        s = S - 1
        path = []
        # Backtracking from T-1 to 0:
        for t in reversed(range(T)):
            path.append(s)
            s = backpointers[t, s].item()
        path = list(reversed(path))
        aligned_targets = [target[s] for s in path]
        return aligned_targets


    def _compute_alignment(self, alignment):
        current_phone = None
        start_time = 0
        aligned_phonemes = []

        for i, phone_id in enumerate(alignment):
            if phone_id == self.blank:
                continue
            phone = self.id_to_phone[phone_id]
            if phone != current_phone:
                if current_phone is not None:
                    aligned_phonemes.append((current_phone, start_time, i))
                current_phone = phone
                start_time = i

        if current_phone is not None:
            aligned_phonemes.append((current_phone, start_time, len(alignment)))

        if aligned_phonemes:
            aligned_phonemes[0] = (aligned_phonemes[0][0], 0, aligned_phonemes[0][2])

        return aligned_phonemes
    

class TextDurationDataset(Dataset):
    """
    Dataset for training a text-based duration predictor.
    Each sample should contain:
      - 'sentence': the input text (to be converted to phoneme sequence)
      - Optionally, 'duration': the ground-truth durations for each phoneme
         (if not present, you could compute them via your ForcedAligner with the audio file).
    """
    def __init__(self, samples, forced_aligner, features_path=None):
        self.samples = samples
        self.forced_aligner = forced_aligner
        self.features_path = features_path

    def __len__(self):
        return len(self.samples)

    def f_read_raw_mat(self, filename, col, data_format="f4", end="l"):
        f = open(filename, "rb")
        if end == "l":
            data_format = "<" + data_format
        datatype = np.dtype((data_format, (col,)))
        data = np.fromfile(f, dtype=datatype)
        f.close()
        if data.ndim == 2 and data.shape[1] == 1:
            return data[:, 0]
        else:
            return data
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Convert text to phoneme ids
        transcription = sample['sentence']
        phonemes = self.forced_aligner.aligner.extract_phonemes(transcription)
        # If ground-truth durations are provided, use them;
        # otherwise you might compute them via ForcedAligner.align() (which requires audio).
        if 'alignments' in sample:
            durations = sample['alignments']
        else:
            # For training, you should have ground-truth durations.
            # Alternatively, if you have audio paths available, compute:
            if self.features_path is None:
                durations, _ = self.forced_aligner.align(sample['path'], phonemes)
            else:
                tokens_filename = os.path.join(self.features_path, "LT_" + os.path.basename(sample['path']).split(".")[0] + ".ppg")
                tokens = self.f_read_raw_mat(tokens_filename, 8)  
                features = self.forced_aligner.aligner.whisper.decode_from_codebook_indices(tokens[None,...])[0]
                durations, _ = self.forced_aligner.align(features, phonemes)

            self.samples[idx]['alignments'] = durations
            # raise ValueError("Ground-truth durations not provided in sample.")
        
        mind = min(len(phonemes), len(durations))
        return {
            'phonemes': torch.tensor(phonemes[:mind], dtype=torch.long),
            'durations': torch.tensor(durations[:mind], dtype=torch.float32)
        }

def collate_text_duration_fn(batch):
    """
    Collate function for variable–length phoneme sequences.
    Pads phoneme sequences and their corresponding duration targets.
    """
    # Sort batch by phoneme length (descending)
    batch.sort(key=lambda x: len(x['phonemes']), reverse=True)
    
    phoneme_seqs = [item['phonemes'] for item in batch]
    duration_seqs = [item['durations'] for item in batch]
    
    phoneme_lens = torch.tensor([len(seq) for seq in phoneme_seqs], dtype=torch.long)
    phoneme_pad = nn.utils.rnn.pad_sequence(phoneme_seqs, batch_first=True, padding_value=0)
    # breakpoint()
    durations_pad = nn.utils.rnn.pad_sequence(duration_seqs, batch_first=True, padding_value=0)
    
    return {
        'phonemes': phoneme_pad.to(Config.DEVICE, non_blocking=True),
        'phonemes_lens': phoneme_lens.to(Config.DEVICE, non_blocking=True),
        'durations': durations_pad.to(Config.DEVICE, non_blocking=True)
    }

class TextDurationPredictor(nn.Module):
    """
    A simple text-based duration predictor.
    The network embeds the phoneme sequence and uses a couple of 1D convolution layers
    to predict a (positive) duration value for each phoneme.
    """
    def __init__(self, num_phonemes, embedding_dim=256, hidden_dim=256, kernel_size=3, dropout=0.1):
        super(TextDurationPredictor, self).__init__()
        self.embedding = nn.Embedding(num_phonemes, embedding_dim)
        # Using two convolution layers to capture local context over the phoneme sequence
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.relu2 = nn.ReLU()
        # Project to one scalar duration per time-step (phoneme)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, phoneme_seq):
        # phoneme_seq: (B, T) where T is the number of phonemes in each sequence
        x = self.embedding(phoneme_seq)         # (B, T, embedding_dim)
        x = x.transpose(1, 2)                   # (B, embedding_dim, T)
        x = self.conv1(x)                       # (B, hidden_dim, T)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)                       # (B, hidden_dim, T)
        x = self.relu2(x)
        x = x.transpose(1, 2)                   # (B, T, hidden_dim)
        durations = self.linear(x).squeeze(-1)   # (B, T)
        # Ensure durations are non-negative (using softplus)
        durations = F.softplus(durations)
        return durations

import os
import time
import torch
import torch.nn as nn

class VATrainer:
    """
    Trainer class for training a text-based duration predictor.
    Includes training and validation loops, logging, and model checkpointing.
    """
    def __init__(self, model, optimizer, criterion, config, forced_aligner,
                 device="cuda", num_epochs=1, log_interval=10, save_path=None):
        """
        Args:
            model (nn.Module): The text duration predictor model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss criterion.
            device (torch.device): Device to run the model on.
            num_epochs (int): Number of epochs to train.
            log_interval (int): Logging frequency (in batches).
            save_path (str, optional): File path to save the best model (by validation loss).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.forced_aligner = forced_aligner
        self.train_loader, self.val_loader = self.prepare_datasets()
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_path = save_path
        self.best_val_loss = float('inf')

    def prepare_datasets(self):
        # Load all datasets
        all_samples = []
        for path in self.config.DATASET_PATHS:
            with open(path) as f:
                all_samples += json.load(f)
                
        # Split datasets
        train_samples, val_samples = train_test_split(
            all_samples, 
            train_size=self.config.TRAIN_RATIO,
            random_state=self.config.RANDOM_SEED
        )
        
        train_dataset = TextDurationDataset(train_samples, self.forced_aligner, features_path=self.config.FEATURES_PATH)
        val_dataset = TextDurationDataset(val_samples[:256], self.forced_aligner, features_path=self.config.FEATURES_PATH)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_text_duration_fn,
            # num_workers=16, 
            # pin_memory=True,
            # persistent_workers=True,
            # prefetch_factor=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            collate_fn=collate_text_duration_fn,
            # num_workers=16, 
            # pin_memory=True,
            # persistent_workers=True,
            # prefetch_factor=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Trains the model for one epoch and logs intermediate losses."""
        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(pbar):
            phonemes = batch['phonemes']  # (B, T)
            durations = batch['durations']  # (B, T)
            
            # Move data to device
            phonemes = phonemes.to(self.device)
            durations = durations.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(phonemes)  # (B, T)
            loss = self.criterion(outputs, durations)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = running_loss / self.log_interval
                elapsed = time.time() - start_time
                print("---------------------------")
                print(outputs[0].detach().cpu().numpy().astype(np.int32))
                print("---------------")
                print(durations[0].detach().cpu().numpy().astype(np.int32))
                print("---------------------------")
                print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Elapsed: {elapsed:.2f}s")
                running_loss = 0.0
                start_time = time.time()
            

    def validate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                phonemes = batch['phonemes'].to(self.device)
                durations = batch['durations'].to(self.device)
                outputs = self.model(phonemes)
                loss = self.criterion(outputs, durations)
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / count if count > 0 else 0.0
        return avg_loss

    def train(self):
        """Runs the full training loop with validation and checkpointing."""
        for epoch in range(self.num_epochs):
            print(f"\nStarting epoch {epoch+1}/{self.num_epochs}")
            self.train_epoch(epoch)
            
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Validation Loss: {val_loss:.4f}")
                
                # Save model if validation loss improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.save_path is not None:
                        torch.save(self.model.state_dict(), self.save_path)
                        print(f"Saved best model to {self.save_path}")
            else:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] completed (no validation set).")







# Example usage:
if 0 and __name__ == '__main__':
    # Assume you have a Config with DEVICE and an "aligner" object available
    # For example:
    #   Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   aligner.phone_to_id exists, and aligner.extract_phonemes(text) returns a list of ids
    from rvqwhisper import RVQFasterWhisperWrapper
    config = Config()
    WW = RVQFasterWhisperWrapper(config_path=os.environ.get("MODELS_DIR", "models") + "/RVQWhisper/config.yaml", 
                                #  rvq_model_path=os.environ.get("MODELS_DIR", "models") + "/RVQWhisper/whisper_largev2_rvq8_fr/rvq_model.pth")
                                 rvq_model_path=os.environ.get("MODELS_DIR", "models") + "/RVQWhisper/whisper_largev2_rvq8_en/rvq_model.pth")
    aligner = Aligner(whisper_wrapper=WW, lang="eng")
    ctcmodel = CTCHead(config.INPUT_DIM, config.NUM_PHONEMES).to(config.DEVICE)
    checkpoint = torch.load(os.environ.get("MODELS_DIR", "models") + "/Aligner/aligner_english.pth")
    ctcmodel.load_state_dict(checkpoint['model_state_dict'])
    forced_aligner = ForcedAligner(ctcmodel, aligner, blank=config.NUM_PHONEMES)

    # Initialize model, optimizer, and loss criterion
    num_phonemes = len(aligner.phone_to_id)
    model = TextDurationPredictor(num_phonemes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Initialize and run trainer
    trainer = VATrainer(model=model,
                        config=config,
                        forced_aligner=forced_aligner,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=Config.DEVICE,
                        num_epochs=2,
                        log_interval=100,
                        save_path="dur_predictor.pt")

    trainer.train()
    # In inference mode, given a text, you can convert it to phoneme sequence and predict durations:
    model.eval()
    with torch.no_grad():
        # Example text input:
        text = "your example sentence here"
        phoneme_seq = aligner.extract_phonemes(text)
        phoneme_tensor = torch.tensor(phoneme_seq, dtype=torch.long).unsqueeze(0).to(Config.DEVICE)
        predicted_durations = model(phoneme_tensor)
        print("Predicted durations:", predicted_durations.cpu().numpy().squeeze())


    
# Usage Example
if 1 and __name__ == "__main__":
    # Initializations

    
    from rvqwhisper import RVQFasterWhisperWrapper
    config = Config()
    WW = RVQFasterWhisperWrapper(config_path=os.environ.get("MODELS_DIR", "models") + "/RVQWhisper/config.yaml", 
                                 rvq_model_path=os.environ.get("MODELS_DIR", "models") + "/RVQWhisper/whisper_largev2_rvq8_en/rvq_model.pth")
    aligner = Aligner(whisper_wrapper=WW, lang="eng")
    trainer = CTCTrainer(config, aligner)
    checkpoint = torch.load(os.environ.get("MODELS_DIR", "models") + "/Aligner/aligner_english.pth")
    trainer.model.load_state_dict(checkpoint['model_state_dict'])

    # Train model
    trainer.train()

    exit()

    # Inference
    model = CTCHead(config.INPUT_DIM, config.NUM_PHONEMES).to(config.DEVICE)
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    forced_aligner = ForcedAligner(model, aligner, blank=config.NUM_PHONEMES)

    # example
    result = forced_aligner.align(
        audio_path="../tmp/tmpfrench.wav",
        target_ids=aligner.extract_phonemes("Est-ce que tu as des frères ou des sœurs ? as tu des frères ou des sœurs ? tu as des frères ou des sœurs ?")
    )
    print(result)

    infer = PhonemeInference(aligner, "checkpoints/best_model.pth")
    bottleneck = aligner.extract_features("../tmp/tmpfrench.wav")[0]
    result = infer.predict(bottleneck)
    print("Predicted phonemes:", result)    
