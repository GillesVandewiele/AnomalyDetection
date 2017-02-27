from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sksos import SOS
from sklearn.neighbors import NearestNeighbors

from analytics.lof import LOF

import numpy as np
import pandas as pd


class Detector(object):
    def __init__(self):
        pass

    def get_name(self):
        raise NotImplementedError('This function needs to be implemented')

    def detect_anomalies(self, data, **params):
        raise NotImplementedError('This function needs to be implemented')


class PCADetector(object):
    def __init__(self, detector):
        self.detector = detector

    def get_name(self):
        raise NotImplementedError('This function needs to be implemented')

    def detect_anomalies(self, data, n_components=3, **params):
        pca = PCA(n_components=n_components)
        return self.detector.detect_anomalies(pca.fit_transform(data), **params)


class TSNEDetector(object):
    def __init__(self, detector):
        self.detector = detector

    def get_name(self):
        raise NotImplementedError('This function needs to be implemented')

    def detect_anomalies(self, data, n_components=3, **params):
        tsne = TSNE(n_components=n_components)
        return self.detector.detect_anomalies(tsne.fit_transform(data), **params)


class IsolationForestDetector(Detector):
    def get_name(self):
        return 'Isolation Forest'

    def detect_anomalies(self, data, **params):
        iso_forest = IsolationForest(verbose=1)
        iso_forest.set_params(**params)
        iso_forest.fit(data)
        return iso_forest.decision_function(data)  # The anomaly score. The lower, the more abnormal.


class EllipticEnvelopeDetector(Detector):
    # Important! assumes gaussian distributions of data
    # Important! assumes that the number of outliers in known in advance (contamination param)
    # Important! n_samples > n_features ** 2  (apply PCA if this is not the case)
    def get_name(self):
        return 'Elliptic Envelope'

    def detect_anomalies(self, data, **params):
        envelope = EllipticEnvelope()
        envelope.set_params(**params)
        envelope.fit(data)
        # TODO: decision function has other range than that of IsolationForest
        return envelope.decision_function(data)  # The anomaly score. The lower, the more abnormal.


class LOFDetector(Detector):
    def get_name(self):
        return 'Local Outlier Factor'

    def detect_anomalies(self, data, **params):
        data = [tuple(x) for x in data.to_records(index=False)]
        lof = LOF(data, normalize=False)
        min_pts = 3
        if 'min_pts' in params:
            min_pts = params['min_pts']
        return [-lof.local_outlier_factor(min_pts, tuple(data[i])) for i in range(len(data))]


class SOSDetector(Detector):
    def get_name(self):
        return 'Stochastic Outlier Selection'

    def detect_anomalies(self, data, **params):
        perplexity = 30
        metric = 'euclidean'
        eps = 1e-5
        if 'perplexity' in params:
            perplexity = params['perplexity']
        if 'metric' in params:
            metric = params['metric']
        if 'eps' in params:
            eps = params['eps']
        sos = SOS(perplexity=perplexity, metric=metric, eps=eps)  # https://github.com/jeroenjanssens/scikit-sos
        if isinstance(data, pd.DataFrame):
            return -sos.predict(data.values)
        else:
            return -sos.predict(data)


class EnsembleDetector(Detector):
    def get_name(self):
        return 'Ensembled detector'

    def detect_anomalies(self, data, **params):
        knn_scores = KNNDetector().detect_anomalies(data)
        sos_scores = SOSDetector().detect_anomalies(data)
        hbos_scores = HBOSDetector().detect_anomalies(data)
        scores = [knn_scores, sos_scores, hbos_scores]

        norm_scores = []
        for score in scores:
            norm_score = np.array(score)
            _max = max(norm_score)
            norm_score /= _max
            norm_scores.append(norm_score)

        return -np.sum(np.array(norm_scores), axis=0)


class HBOSDetector(Detector):
    def get_name(self):
        return 'Histogram-Based Outlier Score'

    def detect_anomalies(self, data, **params):
        # http://www.dfki.de/KI2012/PosterDemoTrack/ki2012pd13.pdf

        k = 3   # How many bins do we use in each histogram?
        if 'k' in params:
            k = params['k']

        if isinstance(data, pd.DataFrame):
            data = data.as_matrix()

        # TODO: Currently, this is fixed-width binning.
        # TODO: Implement variable-width binning (same nr of samples in each bin, see paper)
        histograms = {}
        for i in range(data.shape[1]):
            histograms[i] = np.histogram(data[:, i], bins=k, normed=True)

        scores = []
        for i in range(data.shape[0]):
            record = data[i, :]
            score = 0
            for j in range(len(record)):
                histogram = histograms[j]
                for k in range(len(histogram[1])-1):
                    if histogram[1][k] <= record[j] < histogram[1][k + 1]:
                        score += np.log(histogram[0][k])
                        break
            scores.append(score)

        return scores


class KNNDetector(Detector):
    def get_name(self):
        return 'K-Nearest Neighbors'

    def detect_anomalies(self, data, **params):
        k = 3
        if 'k' in params:
            k = params['k']
        distances, indices = NearestNeighbors(n_neighbors=k+1).fit(data).kneighbors(data)

        return -np.sum(distances, axis=1)