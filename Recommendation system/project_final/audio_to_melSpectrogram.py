#!/usr/bin/env python3

import os
import pandas as pd
import re
import math
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt



sampling_rate = 22050
hop_length = 512 # to make time steps 128
#     fmin = 20
#     fmax = sampling_rate // 2
n_mels = 128
n_fft = 2048


def read_as_melspectrogram(pathname):
    # preprocessing audio to mel spectrogram
    y, sr = librosa.load(pathname, sr=sampling_rate)  
    sgram = librosa.stft(audio,n_fft=conf.n_fft,  hop_length=hop_length,window='hann')  
    mel_spectrogram = librosa.feature.melspectrogram(S=sgram, 
                                                 sr=sampling_rate,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft,
                                                   ,fmax=8000)
    mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mels = mel_spect.astype(np.float32)
    return mels

    
def extract_genere():
# Get Genres and Track IDs from the tracks.csv file
    filename_metadata = "data/tracks.csv"
    tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
    tracks_id_array = tracks.values[: , 0]
    tracks_genre_array = tracks_array[: , 40]
    tracks_id_array = tracks_id_array.tolist()
    tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)
    return tracks_id_array,tracks_genre_array


def build_mel_spectrogram_plot(mel,file_name,mode='Train'):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = float(mel.shape[1]) / float(100)
    fig_size[1] = float(mel.shape[0]) / float(100)
    plt.rcParams["figure.figsize"] = fig_size
    plt.axis('off')
    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel, cmap='gray_r')
    if mode='Train':
        plt.savefig("Train_Spectogram_Images/"+file_name+".jpg", bbox_inches=None, pad_inches=0)
        print('Save Success')
    elif mode='Test':
        plt.savefig("Test_Spectogram_Images/"+file_name+".jpg", cmap='gray_r', bbox_inches=None, pad_inches=0)
    plt.close() 
def create_spectrogram(mode='Train'):
    if mode == "Train":
        tracks_id_array,tracks_genre_array=extract_genere()
        folder_sample = "data/fma_small"
        directories = [d for d in os.listdir(folder_sample) if os.path.isdir(folder_sample + '/' + d)]
        counter = 0
        if not os.path.exists('Train_Spectogram_Images'):
            os.makedirs('Train_Spectogram_Images')
        for d in directories:
            file_names = []
            label_directory = folder_sample + '/' + d
            for f in os.listdir(label_directory):
                if f.endswith(".mp3"):
                    file_names.append(label_directory + '/' + f)
            # Convert .mp3 files into mel-Spectograms
            for f in file_names:
                track_id = int(re.search('fma_small/.*/(.+?).mp3', f).group(1))
                track_index = tracks_id_array.index(track_id)
                if(str(tracks_genre_array[track_index, 0]) != '0'):
                    mel = read_as_melspectrogram(f)
                    # Length and Width of Spectogram
                    file_name=str(counter)+"_"+str(tracks_genre_array[track_index,0])
                    build_mel_spectrogram_plot(mel,file_name,mode)
                    counter = counter + 1

    elif mode == "Test":
        folder_sample = "data/test"
        counter = 0
        if not os.path.exists('Test_Sepctogram_Images'):
            os.makedirs('Test_Spectogram_Images')
        file_names = [folder_sample + '/' + f for f in os.listdir(folder_sample)
                       if f.endswith(".mp3")]
        # Convert .mp3 files into mel-Spectograms
        for f in file_names:
            test_id = re.search('Dataset/DLMusicTest_30/(.+?).mp3', f).group(1)
            mel = read_as_melspectrogram(f)
            build_mel_spectrogram_plot(mel,test_id,mode)
    return
if __name__ == '__main__':
    create_spectrogram(mode='Train')