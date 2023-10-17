import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
import time

import librosa
import torch
import torchaudio
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from encodec.utils import convert_audio
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.optim import Adam
from transformers import AutoProcessor, EncodecModel, EncodecFeatureExtractor
from models import Pose2AudioTransformer
from utils import DanceToMusic
from torch.utils.tensorboard import SummaryWriter

# Define a function to save the model and delete the last saved model
def save_model(model, folder_path, loss, last_saved_model):
    model_name = f"best_model_{loss:.4f}.pt"
    model_path = os.path.join(folder_path, model_name)
    torch.save(model.state_dict(), model_path)
    if last_saved_model:
        try:
            os.remove(os.path.join(folder_path, last_saved_model))
        except FileNotFoundError:
            print(f"Warning: Could not find {last_saved_model} to delete.")
    return model_name


if __name__ == "__main__":

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
    # processor = AutoProcessor.from_pretrained(model_id)
    
    sample_rate = 24000
    batch_size = 16
    # data_dir = "/Users/azeez/Documents/pose_estimation/Learning2Dance/l2d_train_3D"
    data_dir = '/Users/azeez/Documents/pose_estimation/Learning2Dance/youtube_dataset/dataset/samples'
    train_dataset = DanceToMusic(data_dir, encoder = encodec_model, sample_rate = sample_rate, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    src_pad_idx = 0
    trg_pad_idx = 0

    device = torch.device("cpu")
    embed_size = train_dataset.data['poses'].shape[2] * train_dataset.data['poses'].shape[3]
    pose_model = Pose2AudioTransformer(codebook_size, src_pad_idx, trg_pad_idx, device=device, num_layers=4, heads = 8, embed_size=embed_size, dropout=0.1)
    pose_model.to(device)


    learning_rate = 1e-4
    # criterion = CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    # criterion = MSELoss()
    optimizer = torch.optim.Adam(pose_model.parameters(), lr=learning_rate)

    # Set up for tracking the best model
    weights_dir = '/Users/azeez/Documents/pose_estimation/MusicGen/weights'
    best_loss = float('inf')  # Initialize with a high value
    last_saved_model = ''

    writer = SummaryWriter(log_dir='./my_logs')

    num_epochs = 30000
    for epoch in range(num_epochs):
        pose_model.train()
        epoch_loss = 0
        for i, (audio_codes, pose, pose_mask, wav, wav_mask, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Forward pass
            wav = wav.unsqueeze(1)
            pose = pose
            pose_mask = pose_mask
            # encoding = encodec_model.encode(wav, wav_mask)
            # audio_codes = encoding["audio_codes"]
            target = audio_codes.to(device)
            target_for_loss = target[:, :, 1:]

            output_softmax, output_argmax, offset = pose_model(pose.to(device), target.squeeze(1).to(device), src_mask=pose_mask.to(device))

            # Flatten both output and target for computing the loss
            output_softmax = output_softmax[:, offset:, :]
            reshaped_output_softmax = output_softmax.reshape(-1, output_softmax.shape[2]).shape

            # Flatten the target
            # Expected dimension: [24032]
            reshaped_target = target_for_loss.reshape(-1).long()

            # Take the log of softmax output
            # Expected dimension: [16, 1503, 1024]
            log_softmax_output = torch.log(output_softmax)  

            # Flatten the log softmax output for loss computation
            # Expected dimension: [24048, 1024]
            log_softmax_output_reshape = log_softmax_output.view(-1, log_softmax_output.shape[2])  

            # Compute loss
            loss = criterion(log_softmax_output_reshape, reshaped_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # Accumulate loss
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
    
        # Check if this epoch resulted in a better model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            last_saved_model = save_model(pose_model, weights_dir, best_loss, last_saved_model)

        writer.add_scalar('Average Loss', avg_epoch_loss, epoch)

    writer.close()