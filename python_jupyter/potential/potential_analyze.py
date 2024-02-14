import matplotlib.pyplot as plt
import seaborn
import math
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

import plots
import scaler
import fit

def relative_deviation(data_main, data_deviated):
