import pandas as pd
import os

def read_S(path):
    return pd.read_csv(path, header=0, names=['beta', 'S', 'S_err'], delimiter = ' ')