
# Detect and Segment Cough
This repository hosts the codes and models to **detect** and **segment** cough sounds. As the names suggest, detecting a cough returns the probability of a given audio file containing a cough sound. Segment cough returns audio files containing a single segmented cough sound from given an audio file with many cough sounds (output of detect cough). These two methods (detect and segment cough) are the most important building blocks for developing cough-based diagnosis tools such as COVID-19. 


# Input-output 
Input: audio files (.wav) to be predicted to have (multiple) cough sound  
Output: Cough or non-cough (detect), new wav files containing segmented cough

# **Supported Python Version and Model** (IMPORTANT!)
I tested that this model works on Python version >= 3.7.0.  
Less than the version above (e.g., python3.6), the output probability will be "0". The xgboost package must be version 0.90.

# Installation: 

First, install the Python library dependencies in a virtual environment via pip.

```
pip install -r requirements.txt
```

## API/Command Line Usage 
  
```
# Detect cough:
python3 detect_cough.py -i input_file.wav
# Segment cough: 
python3 segment_cough -i input_file.wav
```
 
## Python Usage
``` 
# detect cough, see notebooks for segment cough
import sys
sys.path.append('./src')
from src.feature_class import features
from src.DSP import classify_cough
from scipy.io import wavfile
import pickle

input_file = './sample_recordings/cough.wav'
model = pickle.load(open('./models/cough_classifier', 'rb'))
scaler = pickle.load(open('./models/cough_classification_scaler', 'rb'))

fs, x = wavfile.read(input_file)
prob = classify_cough(x, fs, model, scaler)
print(f"{input_file} has probability of cough: {prob}")
```
# Example
```
# detect cough:
bagus@L140MU:detect-cough$ /usr/bin/python3 detect_cough.py -i sample_recordings/cough.wav
/home/bagus/.local/lib/python3.8/site-packages/sklearn/base.py:329: UserWarning: Trying to unpickle estimator LabelEncoder from version 0.22.1 when using version 1.0.2. This might lead to breaking code or invalid results. Use at your own risk. For more info, please refer to:
https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
  warnings.warn(
/home/bagus/.local/lib/python3.8/site-packages/sklearn/base.py:329: UserWarning: Trying to unpickle estimator StandardScaler from version 0.22.1 when using version 1.0.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
  warnings.warn(
sample_recordings/cough.wav has probability of cough: 0.988208472728729
# segment cough
bagus@L140MU:detect-cough$ ./segment_cough.py -i sample_recordings/cough.wav
Write to ./cough-0.wav
Write to ./cough-1.wav
```

# Overview: 

In the wake of the COVID-19 pandemic, mass coronavirus testing has proven essential to governments in monitoring the spread of the disease, isolating infected individuals, and effectively “flattening the curve” of infections over time. However, this oropharyngeal swab test is physically invasive and must be performed by a trained clinician. This requires patients to travel to a laboratory facility to get tested, thereby potentially infecting others along the way. Ideally, testing would be performed noninvasively at no cost, and administered at the homes of potential patients to minimize contamination risk.

The World Health Organization (WHO) has reported that 67.7% of COVID-19 patients exhibit a “dry cough,” which may be audibly different from coughs caused by other pathologies. Such cough sounds analysis has proven successful in diagnosing respiratory conditions like pertussis, asthma, and pneumonia.

At the Embedded Systems Laboratory (ESL) at EPFL, we have developed the COUGHVID database, which is an extensive dataset of COVID-19 cough sounds from around the world, partially validated by expert pulmonologists. We contribute our data, signal preprocessing source code, cough classification algorithm, and feature extraction methods to assist the global research community in developing algorithms to automatically screen for COVID-19 based on cough sounds.

# Data access

The COUGHVID dataset can be downloaded from the following Zenodo link: https://doi.org/10.5281/zenodo.4048311

## Notebooks
The `coughvid_classification_example.ipynb` notebook illustrates the usage of the cough classifier model for removing unwanted recordings from a cough database.

The `segmentation_and_SNR_example.ipynb` notebook is an example of how to use the automatic cough segmentation and SNR estimation algorithm.

## Source code

### Convert files

A quick function to automatically convert all of the compressed .webm and .ogg files in the COUGHVID dataset to the more usable .wav format. Note: you must have FFMPEG installed for this to work. 

### DSP

This file contains all-digital signal processing functions, including filtering the recordings and classifying between cough sounds and non-cough sounds.

### Features

This file contains all of the functions used for the computation of audio signal features commonly used in cough classification.

### Segmentation

This file contains a function for segmenting a recording into individual cough signals and additional code to compute the SNR of the recording.

## Models

The  `cough_classifier` is an XGB model that can be loaded and used in the `classify_cough` function to classify whether or not a given recording contains cough sounds. The `cough_classification_scaler` is a feature scaler also used in this function.


# Citation 

When using this resource, please cite the following publication: 

Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Sci Data* **8,** 156 (2021). https://doi.org/10.1038/s41597-021-00937-4

# Reference  
1. https://c4science.ch/diffusion/10770/  (original repo forked from)

# Changealog  
- 21/03/2022: Submit paper to interspeech about cough segmentation, commit: f330c2fb90431c736ee495b668ac0b0e0994b0cf  
- 08/02/2022: Rename repo from `detect-cough` to `detect-segment-cough`
- 03/02/2022: Initial version, forked from EPFL's original repo
