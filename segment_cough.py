#!/usr/bin/env python3
""" program to segment cough from audio files """
# bagus@ep.its.ac.id

import librosa
import os
import sys
sys.path.append('./src')
from segmentation import segment_cough
import soundfile as sf
import argparse

def main(input_file, dir_output='./', fs_out=16000):
    x, fs = librosa.load(input_file, sr=fs_out)
    cough_segments, cough_mask = segment_cough(x, fs, cough_padding=0)
    for i in range(0, len(cough_segments)):
        sf.write(dir_output 
                    + os.path.basename(input_file).split('.')[0] 
                    + '-' + str(i) 
                    + '.wav', 
                 cough_segments[i], 
                 fs
        )
        print(f"Write to {dir_output}{os.path.basename(input_file).split('.')[0]}-{str(i)}.wav")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segment cough from audio files')
    parser.add_argument('-i', '--input_file', help='input file')
    parser.add_argument('-o', '--output_dir', help='output directory', 
                        default='./', type=str)
    parser.add_argument('-fs', '--fs_out', help='output sampling rate', 
                        default=16000, type=int)
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.fs_out)