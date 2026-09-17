import torch
class Config:
    # Dataset configuration
    DATASET_PATHS = ["data1.json",
                     "data2.json"]
    PREFS = ["pr1_", "pr2_"]
    FEATURES_PATH = "./Whisper_ppg"
    TRAIN_RATIO = 0.9
    RANDOM_SEED = 42
    
    # Model parameters
    INPUT_DIM = 1280  # Should match whisper bottleneck dimension
    NUM_PHONEMES = 111  # Should match phone_to_id size
    
    # Training parameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 1
    LEARNING_RATE = 3e-4
    PATIENCE = 1
    
    # System
    CHECKPOINT_DIR = "./checkpoints"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"