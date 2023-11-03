# Dance to Music

## Introduction
Welcome to Dance to Music, an iinovative project that blends computer vison, audio signal processing and generative AI to create novel music based on a sequence of human pose estimates. 

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Credits](#credits)
- [License](#license)

## Motivation
Inspried by the recent 'text-to-image' (DALL-E, Midjourney, Stable Diffusion) and 'text-to-music' models (MusicLM, MusicGen), this project aims to develop 'Dance-to-Music'. As input, this model takes in a 5 sec video, and in return produces a 5 second peice of audio that corresponds with the dancer of the video. 

## Features
- Video input analysis for human pose estimation.
- Generation of music through a sequence-to-sequence transformer model.
- Novel audio generation from predicted latent space representations.

## Technologies Used
- [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) for 3D human pose estimation
- [Meta's EnCodec model](https://huggingface.co/docs/transformers/main/model_doc/encodec#transformers.EncodecModel) used to create audio encodings from .wav files, as well as a decoder that reconstructs audio from the encoded representation
- Python, PyTorch

## Installation
One can start playing with this project by first cloneing the repository, building the dataset, and then running the training script. 
```bash
# Clone the repository
git clone https://github.com/azeezabdikarim/DanceToMusic.git

# Navigate to the repository
cd DanceToMusic

# Build the Conda environment and download all necessary pages
conda env create -f environment.yml

# Activate the new Conda environemnt 
conda activate dance2music

# Build the video dataset, extracting 5 second clips at 24fps
# FYI: The video downloads might take a while depending on your internet speeds. Also, mediapipe uses the CPU to calculate human pose estimates, so that will take a decent amount of time. It takes me between 1-2 hours to build the complete dataset of ~3200 clips on a Macbook Pro with an M2 Max chip
python data/building_tools/build_complete_dataset.py --output_path data/samples/ --input_csv data/youtube_links/youtube_links_test.csv  --max_seq_len 5 --fps 24

```
## Usage

## Dataset

## Model Architecture 

## Results

## Credits

## License
