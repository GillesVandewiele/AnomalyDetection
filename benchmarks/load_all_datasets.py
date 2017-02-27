"""
Uses python introspection to call all function in `data.load_datasets`

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

from inspect import getmembers, isfunction
import benchmarks.load_datasets




def load_all_datasets():
    """
    Uses python introspection to call all function in `data.load_datasets`

    **Returns**
    -----------
    a list of loaded datasets
    """
    datasets = []
    for o in getmembers(benchmarks.load_datasets):
        if isfunction(o[1]):
            df, feature_cols, label_col, name = o[1]()
            datasets.append({'dataframe': df, 'feature_cols': feature_cols, 'label_col': label_col, 'name': name})

    return datasets