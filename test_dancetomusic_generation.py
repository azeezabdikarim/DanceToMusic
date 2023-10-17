import os
import numpy as np
import torch
import pickle
import time

import torch
from torch.utils.data import DataLoader, Dataset
from models import Pose2AudioTransformer
from transformers import EncodecModel
from utils import DanceToMusic
from datetime import datetime
from torch.optim import Adam

# assign GPU or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
device = torch.device("cpu")

model_id = "facebook/encodec_24khz"
encodec_model = EncodecModel.from_pretrained(model_id)
codebook_size = encodec_model.quantizer.codebook_size
sample_rate = 24000

data_dir = "/Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/dataset/samples"
dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device)

src_pad_idx = 0
trg_pad_idx = 0
learned_weights = '/Users/azeez/Documents/pose_estimation/MusicGen/weights/first_yout_dataset_.011.pth' 
device = torch.device("mps")
embed_size = dataset.data['poses'].shape[2] * dataset.data['poses'].shape[3]
pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, device=device, num_layers=4, heads = 8, embed_size=embed_size, dropout=0.1)
pose_model.load_state_dict(torch.load(learned_weights, map_location=device))
pose_model.to(device)

audio_codes, pose, pose_mask, wav, wav_mask, _, _ = dataset[0]
output = pose_model.generate(pose.unsqueeze(0).to(device), pose_mask.to(device), max_length = 50)

print(output[0].shape, audio_codes[0].shape)