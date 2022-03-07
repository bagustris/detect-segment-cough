# simple cough segmentation on virufy data
# reference code: 
# https://github.com/zanjabil2502/Tugas_Akhir/blob/main/code/Automatic%20Segmentation%20for%20Cough.ipynb
import librosa

positif = '/home/bagus/github/virufy-data/clinical/original/pos/'
negatif = '/home/bagus/github/virufy-data/clinical/original/neg/'

seg_pos = '/home/bagus/github/virufy-data/clinical/simple/pos/'
seg_neg = '/home/bagus/github/virufy-data/clinical/simple/neg/'


def simple_segment(audio_input)
    wav, fs = librosa.load(audio_input, sr=None)
    rms = lb.feature.rms(y=wav_negative[0])
    times = lb.times_like(rms, sr=fs)
    rms = rms.reshape(rms.shape[1],rms.shape[0])
    rms_norm = Scaler.fit_transform(rms)

    Scaler = MinMaxScaler(feature_range=(0,1))

    return start, end

def segment_dir(dir_input, dir_output):
    for file in os.listdir(dir_input):
        if file.endswith('.wav'):
            simple_segment(file)
            print('Segmented ' + file)