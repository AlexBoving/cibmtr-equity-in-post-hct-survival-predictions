import pandas as pd
import numpy as np
from lifelines.utils import concordance_index

def race_stratified_c_index(y_true, y_pred, events, races):
    unique_races = races.unique()
    c_indices = []
    
    for race in unique_races:
        mask = races == race
        if sum(mask) > 1:  # Ensure at least two samples for comparison
            c_idx = concordance_index(y_true[mask], -y_pred[mask], events[mask])
            c_indices.append(c_idx)
    
    if len(c_indices) > 1:
        return np.mean(c_indices) - np.std(c_indices)
    return np.nan