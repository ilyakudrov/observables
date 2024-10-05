import pandas as pd
import numpy as np
from scipy import integrate

def generate_bootstrap(df, n):
    """
    Generate n random samples distributed normaly
    for each data point with mean and std
    """
    df = df.reset_index()
    samples = np.random.default_rng().normal(df.loc[0, 'S'], df.loc[0, 'S_err'], n)
    return pd.DataFrame(data={'S': samples, 'n': np.arange(1, n + 1)})

def integrate_S(df, function):
    return pd.DataFrame({'energy': function(df['S'], x = df['beta']), 'beta': df.reset_index().loc[1:,'beta']})