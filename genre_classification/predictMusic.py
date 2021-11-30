# We can pick up a certain MP3 music from our local computer or from the dataset
# Do feature extraction and statistical computation
# Put feature into model and give the predicted genre of the music
import os
from keras.models import load_model
from scipy import stats
import utils
import numpy as np
import librosa
import librosa.display

# user parameter
choose_music = 'local'  # 'local' or 'dataset'
# environment
os.environ['AUDIO_DIR'] = '../data/fma_small/'  # if you choose music from database
filename = './testMusic.mp3'  # if you choose local music


# Statistical Computation
def feature_stats(values):
    kurtosis = stats.kurtosis(values, axis=1)
    max = np.max(values, axis=1)
    mean = np.mean(values, axis=1)
    median = np.median(values, axis=1)
    min = np.min(values, axis=1)
    skew = stats.skew(values, axis=1)
    std = np.std(values, axis=1)
    feature = np.array([kurtosis, max, mean, median, min, skew, std])
    feature = feature.T.reshape(1, feature.shape[0] * feature.shape[1])
    return feature


# feature extraction
def feature_extraction(x, sr):
    # zcr
    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    zcr = feature_stats(f)

    # cqt
    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                             n_bins=7 * 12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    chroma_cqt = feature_stats(f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    chroma_cens = feature_stats(f)

    f = librosa.feature.tonnetz(chroma=f)
    tonnetz = feature_stats(f)

    # stft
    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
    del x

    f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
    chroma_stft = feature_stats(f)
    f = librosa.feature.rms(S=stft)
    rmse = feature_stats(f)

    f = librosa.feature.spectral_centroid(S=stft)
    spectral_centroid = feature_stats(f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    spectral_bandwidth = feature_stats(f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    spectral_contrast = feature_stats(f)
    f = librosa.feature.spectral_rolloff(S=stft)
    spectral_rolloff = feature_stats(f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    del stft

    # mfcc
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc = feature_stats(f)

    feature = [chroma_cens, chroma_cqt, chroma_stft, mfcc, rmse, spectral_bandwidth, spectral_centroid,
               spectral_contrast, spectral_rolloff, tonnetz, zcr]
    return feature


# flatten feature into 1 row array
def flat(a):
    l = []
    for i in a:
        for j in i:
            for k in j:
                l.append(k)
    return (l)


# main

print('Music Information:')
# choose directory where mp3 are stored
if choose_music == 'dataset':
    # choose music from dataset
    AUDIO_DIR = os.environ.get('AUDIO_DIR')
    filename = utils.get_audio_path(AUDIO_DIR, 2)

elif choose_music == 'local':
    # choose local music
    filename = filename

print('Music File: {}'.format(filename))

# load mp3 music
x, sr = librosa.load(filename, sr=None, mono=True)
print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

# get feature
feature = feature_extraction(x, sr)

# flatten feature
temp = flat(feature)
length = len(temp)
temp = np.array(temp)
temp = temp.reshape(1, length)
temp = temp.reshape((temp.shape[0], temp.shape[1], 1))

# load model
onedCNN_model = load_model('oneDCNN.h5')
onedCNN_model.summary()

# predict the genre of this music
label_prediction = onedCNN_model.predict(temp)
print('\nthe result after prediction:\n',label_prediction)


genre_name=['Blues','Classical','Country','Easy Listening','Electronic',
            'Experimental','Folk','Hip-Hop','Instrumental','International',
            'Jazz','Old-Time / Historic','Pop','Rock','Soul-RnB','Spoken']

print('\nthe most possible genre is:')
# find the most possibility genre
max_possibility=np.max(label_prediction)
# find the index
label_prediction=label_prediction.tolist()
index=0
for result in label_prediction:
    for i in result:
        if i==max_possibility:
            # print(index)
            print(genre_name[index],' ')
        index+=1



