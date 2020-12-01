import numpy as np
import pandas as pd
import os
import subprocess
from pathlib import Path


def convert_files(folder):
    """Convert files from .webm and .ogg to .wav
    folder: path to coughvid database and metadata_compiled csv"""
    
    df = pd.read_csv(folder + 'metadata_compiled.csv')
    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if (counter%1000 == 0):
            print("Finished {0}/{1}".format(counter,len(names_to_convert)))
        if os.path.isfile(folder + name + '.webm'):
            subprocess.call(["ffmpeg", "-i", folder+name+".webm", folder+name+".wav"])
        elif os.path.isfile(folder + name + '.ogg'):
            subprocess.call(["ffmpeg", "-i", folder+name+".ogg", folder+name+".wav"])
        else:
            print("Error: No file name {0}".format(name))
            
    