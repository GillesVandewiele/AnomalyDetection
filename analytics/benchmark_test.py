import itertools
import resource
import matplotlib.pyplot as plt
import pandas as pd
from benchmarks.load_all_datasets import load_all_datasets
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from analytics.detector import IsolationForestDetector, SOSDetector, KNNDetector, HBOSDetector


resource.setrlimit(
    resource.RLIMIT_CORE,
    (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


detectors = {'Isolation Forest': IsolationForestDetector(), 'Stochastic Outlier Selection': SOSDetector(),
             'K-Nearest Neighbor': KNNDetector(),
             'Histogram Based Outlier Detection': HBOSDetector()}
params_per_detector = {}   # Here we can store the parameters for each of the detectors

fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()


for dataset in load_all_datasets():
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    _name = dataset['name']

    for detector in detectors:
        print(detectors[detector].get_name(), _name)
        scores = detectors[detector].detect_anomalies(df[feature_cols])
        df[detector] = scores
        plt.title('Distribution of '+detectors[detector].get_name()+' scores for inliers and outliers on '+_name)
        plt.hist(list(itertools.compress(df[detector], df['Class'].values == 0)), normed=True, alpha=0.5,
                 label='outliers')
        plt.hist(list(itertools.compress(df[detector], df['Class'].values == 1)), normed=True, alpha=0.5,
                 label='inliers')
        plt.xlabel(detector+' Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        # Outliers must have a low score, while inliers must have a high score!
        fpr[detector], tpr[detector], thresholds[detector] = roc_curve(df['Class'].values, df[detector], pos_label=1)
        roc_auc[detector] = auc(fpr[detector], tpr[detector])

        # pca_scores = PCADetector(detectors[detector]).detect_anomalies(df[feature_cols])
        # df['PCA_'+detector] = pca_scores

        # plt.title('Distribution of PCA + ' + detector + ' scores for inliers and outliers')
        # plt.hist(list(itertools.compress(df['PCA_'+detector], df['Class'].values == 0)), normed=True, alpha=0.5,
        #          label='outliers')
        # plt.hist(list(itertools.compress(df['PCA_'+detector], df['Class'].values == 1)), normed=True, alpha=0.5,
        #          label='inliers')
        # plt.xlabel('PCA_'+detector + ' Score')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()

        # fpr['PCA_'+detector], tpr['PCA_'+detector], thresholds['PCA_'+detector] = roc_curve(df['Class'].values, df['PCA_'+detector])
        # roc_auc['PCA_'+detector] = auc(fpr['PCA_'+detector], tpr['PCA_'+detector])
        #
        # tsne_scores = TSNEDetector(detectors[detector]).detect_anomalies(df[feature_cols])
        # df['TSNE_' + detector] = tsne_scores

        # plt.title('Distribution of TSNE + ' + detector + ' scores for inliers and outliers')
        # plt.hist(list(itertools.compress(df['TSNE_' + detector], df['Class'].values == 0)), normed=True, alpha=0.5,
        #          label='outliers')
        # plt.hist(list(itertools.compress(df['TSNE_' + detector], df['Class'].values == 1)), normed=True, alpha=0.5,
        #          label='inliers')
        # plt.xlabel('TSNE_' + detector + ' Score')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()

        # fpr['TSNE_' + detector], tpr['TSNE_' + detector], thresholds['TSNE_' + detector] = roc_curve(df['Class'].values,
        #                                                                                           df['TSNE_' + detector])
        # roc_auc['TSNE_' + detector] = auc(fpr['TSNE_' + detector], tpr['TSNE_' + detector])

    plt.figure()
    lw = 2
    colors = "bgrcmyk"
    for i, detector in enumerate(detectors):
        plt.plot(fpr[detector], tpr[detector], color=colors[i],
                 lw=lw, label='ROC curve '+detectors[detector].get_name()+' (area = %0.2f)' % roc_auc[detector])
        # plt.plot(fpr['PCA_'+detector], tpr['PCA_'+detector], color=colors[i],
        #          lw=lw, label='ROC curve PCA+'+detectors[detector].get_name()+' (area = %0.2f)' % roc_auc['PCA_'+detector])
        # plt.plot(fpr['TSNE_'+detector], tpr['TSNE_'+detector], color=colors[i],
        #          lw=lw, label='ROC curve TSNE+'+detectors[detector].get_name()+' (area = %0.2f)' % roc_auc['TSNE_'+detector])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic '+_name)
    plt.legend(loc="lower right")
    plt.show()


features_df = pd.read_csv('../data/sample_features.csv')
features_columns = list(set(features_df.columns) - {'UserId'})
features_df[features_df > 100000] = 10

print(features_df.values.shape)