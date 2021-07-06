
# COUGHVID: A cough-based COVID-19 fast screening project

# Overview: 

In the wake of the COVID-19 pandemic, mass coronavirus testing has proven essential to governments in monitoring the spread of the disease, isolating infected individuals, and effectively “flattening the curve” of infections over time. However, this oropharyngeal swab test is physically invasive and must be performed by a trained clinician. This requires patients to travel to a laboratory facility to get tested, thereby potentially infecting others along the way. Ideally, testing would be performed noninvasively at no cost, and administered at the homes of potential patients to minimize contamination risk.

The World Health Organization (WHO) has reported that 67.7% of COVID-19 patients exhibit a “dry cough,” which may be audibly different from coughs caused by other pathologies. Such cough sounds analysis has proven successful in diagnosing respiratory conditions like pertussis, asthma, and pneumonia.

At the Embedded Systems Laboratory (ESL) at EPFL, we have developed the COUGHVID database, which is an extensive dataset of COVID-19 cough sounds from around the world, partially validated by expert pulmonologists. We contribute our data, signal preprocessing source code, cough classification algorithm, and feature extraction methods to assist the global research community in developing algorithms to automatically screen for COVID-19 based on cough sounds.

# Data access

The COUGHVID dataset can be downloaded from the following Zenodo link: https://doi.org/10.5281/zenodo.4048311

# How to use it: 

First install the Python library dependencies in a virtual environment.

Pip:

```
pip install -r requirements.txt
```

Conda:

```
conda env create -f environment.yml
```

## Notebooks
The `coughvid_classification_example.ipynb` notebook illustrates the usage of the cough classifier model for removing unwanted recordings from a cough database.

The `segmentation_and_SNR_example.ipynb` notebook is an example of how to use the automatic cough segmentation and SNR estimation algorithm.

## Source code

### Convert files

A quick function to automatically convert all of the compressed .webm and .ogg files in the COUGHVID dataset to the more usable .wav format. Note: you must have FFMPEG installed for this to work. 

### DSP

This file contains all digital signal processing functions, including filtering the recordings and classifying between cough and non-cough sounds.

### Features

This file contains all of the functions used for the computation of audio signal features commonly used in cough classification.

### Segmentation

This file contains a function for segmenting a recording into individual cough signals, and additional code to compute the SNR of the recording.

## Models

The  `cough_classifier` is an XGB model that can be loaded and used in the `classify_cough` function to classify whether or not a given recording contains cough sounds. The `cough_classification_scaler` is a feature scaler also used in this function.


# Citation

When using this resource, please cite the following publication: 

Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Sci Data* **8,** 156 (2021). https://doi.org/10.1038/s41597-021-00937-4

# Contact

For questions or suggestions, please contact coughvid@epfl.ch

To donate a COVID-19 cough sound to our database, please visit https://coughvid.epfl.ch/