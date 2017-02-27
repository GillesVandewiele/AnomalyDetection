"""Contains data set loading functions. If you want the test script to include a new dataset, a new function must
be written in this module that returns a pandas Dataframe, the feature column names, the label column name and the
dataset name.

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""
from collections import Counter

from sklearn import datasets

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def load_wisconsin_breast_cancer():
    """
    The Breast Cancer Wisconsin (Original) dataset from UCI machine learning repository is a classification dataset,
    which records the measurements for breast cancer cases. There are two classes, benign and malignant.
    This dataset has dimensionality 9. The malignant class of this dataset is considered as outliers (239 (35%)),
    while points in the benign class are considered inliers.
    """

    columns = ['ID', 'ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize',
               'BareNuclei',
               'BlandChromatin', 'NormalNuclei', 'Mitoses', 'Class']
    features = ['ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
                'BlandChromatin', 'NormalNuclei', 'Mitoses']
    df = pd.read_csv(
        os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'breast-cancer-wisconsin.data'),
        header=None)
    df.columns = columns
    df['Class'] = np.subtract(np.divide(df['Class'], 2), 1)
    df['Class'] = df['Class'].map({0: 1, 1: 0})
    df = df.drop('ID', axis=1).reset_index(drop=True)
    df['BareNuclei'] = df['BareNuclei'].replace('?', int(np.mean(df['BareNuclei'][df['BareNuclei'] != '?'].map(int))))
    df = df.applymap(int)

    return df, features, 'Class', 'wisconsinBreast'


def load_lympho():
        """
        The original lymphography dataset from UCI machine learning repository is a classification dataset.
        It is a multi-class dataset having four classes, but two of them are quite small (2 and 4 data records).
        Therefore, those two small classes are merged and considered as outliers compared to other two large classes.
        """
        df = pd.read_csv(
            os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'lymphography.dat'))
        features = list(set(df.columns) - {'Class'})
        df['Class'] = df['Class'].map({'fibrosis': 0, 'normal': 0, 'metastases': 1, 'malign_lymph': 1})
        df = MultiColumnLabelEncoder().fit_transform(df)
        return df, features, 'Class', 'lympho'


def load_cardio():
    """
    The original Cardiotocography (Cardio) dataset from UCI machine learning repository consists of measurements of
    fetal heart rate (FHR) and uterine contraction (UC) features on cardiotocograms classified by expert obstetricians.
    This is a classification dataset, where the classes are normal, suspect, and pathologic. For outlier detection,
    The normal class formed the inliers, while the pathologic (outlier) class is downsampled to 176 points.
    The suspect class is discarded.
    """
    df = pd.read_csv(
        os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'ctg.csv'))
    df = df.dropna(axis=1, how='all')
    df = df.dropna()
    df['NSP'] = np.subtract(df['NSP'], 1)
    df = df[df['NSP'] <= 1]
    df['Class'] = df['NSP'].map({0: 1, 1:0})
    df = df.drop('NSP', axis=1)
    features = list(set(df.columns) - {'NSP'})

    return df, features, 'Class', 'cardio'


def load_arrhythmia():
    """
    The original arrhythmia dataset from UCI machine learning repository is a multi-class classification dataset with
    dimensionality 279. There are five categorical attributes which are discarded here, totalling 274 attributes.
    The smallest classes, i.e., 3, 4, 5, 7, 8, 9, 14, 15 are combined to form the outliers class and
    the rest of the classes are combined to form the inliers class.
    """
    df = pd.read_csv(
        os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'arrhythmia.data'),
        header=None)

    df = df.apply(pd.to_numeric, args=('coerce',))

    df = df.dropna(axis=1, how='any')
    df = df.dropna()

    y = df.iloc[:, -1]  # The last column is the ground-truth label vector
    y.name = 'Class'
    y = y.map({3: 0, 4: 0, 5: 0, 7: 0, 8: 0, 9: 0, 14: 0, 15: 0, 0: 1, 1: 1, 2: 1,
               6: 1, 10: 1, 11: 1, 12: 1, 13: 1, 16: 1})

    X = df.iloc[:, :-1]  # The first to second-last columns are the features

    return pd.concat([X, y], axis=1), list(X.columns), y.name, 'Arrhythmia'



# def load_shuttle():
#     """
#     The original Statlog (Shuttle) dataset from UCI machine learning repository is a multi-class classification dataset
#     with dimensionality 9. Here, the training and test data are combined.
#     The smallest five classes, i.e. 2, 3, 5, 6, 7 are combined to form the outliers class,
#     while class 1 forms the inlier class. Data for class 4 is discarded.
#     """
#
#     columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
#                'feature9', 'Class']
#     features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
#                 'feature9']
#
#     df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'shuttle.trn'),
#                      sep=' ')
#     df.columns = columns
#     for feature in features:
#         if np.min(df[feature]) < 0:
#             df[feature] += np.min(df[feature]) * (-1)
#     df = df[df['Class'] != 4]  # Data for class 4 is discarded.
#     df['Class'] = df['Class'].map({1: 1, 2: 0, 3: 0, 5: 0, 6: 0, 7: 0})  # 2, 3, 5, 6, 7 are combined
#     df = df.reset_index(drop=True)
#     return df, features, 'Class', 'shuttle'
