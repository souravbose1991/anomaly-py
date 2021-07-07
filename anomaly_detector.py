
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from kneed import KneeLocator

from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


class Detector:

    def __init__(self, data, features, lib='pyod', algo='IForest', impurity=0.05, **kwargs):
        
        if lib=='pyod':
            self.lib = lib
            if impurity == 'auto':
                raise ValueError("PYOD: Impurity can be float in range (0, 0.5)")
            self.impurity = impurity
            self.classifiers = {
                'ABOD': ABOD(contamination=self.impurity, method='fast'),
                'CBLOF': CBLOF(contamination=self.impurity, check_estimator=False, random_state=np.random.RandomState(42)),
                'HBOS': HBOS(contamination=self.impurity),
                'IForest': IForest(contamination=self.impurity, random_state=np.random.RandomState(42)),
                'KNN': KNN(contamination=self.impurity),
                'Avg-KNN': KNN(method='mean', contamination=self.impurity),
                'LOF': LOF(n_neighbors=35, contamination=self.impurity),
                'MCD': MCD(contamination=self.impurity, random_state=np.random.RandomState(42)),
                'OCSVM': OCSVM(contamination=self.impurity),
                'PCA': PCA(contamination=self.impurity, random_state=np.random.RandomState(42))
            }
            if algo not in self.classifiers.keys():
                raise ValueError("PYOD-Algorithm should be one of the following:\n" + str(self.classifiers.keys()))

        elif lib=='sklearn':
            self.lib = lib
            self.impurity = impurity
            self.classifiers = {
                'IForest': IsolationForest(contamination=self.impurity, random_state=np.random.RandomState(42)),
                'LOF': LOF(n_neighbors=35, contamination=self.impurity),
                'OCSVM': OCSVM(contamination=self.impurity)
            }
            if algo not in self.classifiers.keys():
                raise ValueError("PYOD-Algorithm should be one of the following:\n" + str(self.classifiers.keys()))
            
        else:
            raise ValueError("lib should be either 'pyod' or 'sklearn'")

        self.algo = algo
        for item in features:
            if item not in data.columns:
                raise ValueError("Feature: " + str(item) + " not found in the data" )

        self.data = data
        self.features = features
        self.kwargs = kwargs

    
    def __suggest_knee(self, curve='convex', direction='decreasing'):
        sense_rat = [0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
        y_values = [len(self.data.loc[self.data['anomaly_score'] >= float(cutoff/100)]) for cutoff in range(5, 100, 5)]
        norm_y = [float(y/len(self.data)) for y in y_values]
        x_values = [float(cutoff/100) for cutoff in range(5, 100, 5)]
        sensitivity = [int(s*len(y_values)) for s in sense_rat]

        # hold knee points for each sensitivity
        knees = []
        norm_knees = []

        # S should be chosen as a function of how many points are in the system (n)
        for s in sensitivity:
            kl = KneeLocator(x_values, y_values, curve=curve, direction=direction, S=s)
            knees.append(kl.knee)
            norm_knees.append(kl.norm_knee)
        
        fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines+markers',
                                        text=[f'Anomaly Share: {round(yval*100,0)}%' for yval in norm_y], hoverinfo='text'))
        # for item in knees:
        #     fig.add_shape(type='line', x0=item, y0=0, x1=item, line=dict(width=3, dash='dashdot'))
        
        fig.update_layout(title='Score Distribution', xaxis_title='Anomaly Score', yaxis_title='# Data points')
        fig.show()
        return knees


    def anomaly_score(self, return_data=True):
        clf = self.classifiers[self.algo]
        X = self.data[self.features]

        if self.lib=='pyod':
            clf.fit(X)
            out_scr = clf.predict_proba(X, method='linear')
            self.data['anomaly_score'] = out_scr[:, 1]

        elif self.lib=='sklearn':
            pass

        # Score Plot
        knees = self.__suggest_knee(curve='convex', direction='decreasing')
        if return_data:
            return self.data


    def top_features(self, threshold=0.5, n_features=10, return_features=False):
        self.data['anomaly'] = np.where(self.data['anomaly_score'] >= threshold, 1, 0)
        rf = RandomForestClassifier()
        X = self.data[self.features]
        rf.fit(X, self.data['anomaly'])

        feature_importances = pd.DataFrame(rf.feature_importances_, index=self.data[self.features].columns,
                                           columns=['importance']).sort_values('importance', ascending=False)

        top_feat_name = list(feature_importances.index)[0:n_features]
        top_feat_value = feature_importances['importance'].tolist()[0:n_features]
        fig = go.Figure(go.Bar(x=top_feat_value, y=top_feat_name, orientation='h'))
        fig.show()
        if return_features:
            return top_feat_name


    def visualize(self, threshold=0.5, dimensions=2, n_features=10, return_data=True):
        feats = self.top_features(threshold=threshold, n_features=n_features, return_features=True)
        
        outliers = self.data.loc[self.data['anomaly'] == 1]
        inliers = self.data.loc[self.data['anomaly'] == 0]
        outliers_sample1 = outliers.sample(n=min(200, outliers.shape[0]), replace=False, random_state=42)
        inliers_sample1 = inliers.sample(n=min(800, inliers.shape[0]), replace=False, random_state=42)
        sample_data = outliers_sample1.append(inliers_sample1, ignore_index=True)
        
        X = sample_data[feats]
        x_transformed = manifold.MDS(n_components=dimensions, metric=True, n_init=1, max_iter=300,
                                     dissimilarity='euclidean').fit_transform(X)
        if dimensions==2:
            self.data_mds = pd.DataFrame(x_transformed, columns=['Dim-1', 'Dim-2'])
        elif dimensions == 3:
            self.data_mds = pd.DataFrame(x_transformed, columns=['Dim-1', 'Dim-2', 'Dim-3'])
        else:
            raise ValueError("Choose dimensions as 2 or 3")

        self.data_mds['anomaly'] = sample_data['anomaly']

        outliers_sample = self.data_mds.loc[self.data_mds['anomaly'] == 1]
        inliers_sample = self.data_mds.loc[self.data_mds['anomaly'] == 0]

        if dimensions == 2:
            fig = go.Figure(data=go.Scatter(x=outliers_sample['Dim-1'], y=outliers_sample['Dim-2'], 
                                            mode='markers', name="Anomaly", marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=inliers_sample['Dim-1'], y=inliers_sample['Dim-2'],
                                     mode='markers', name="Normal",  marker=dict(size=5)))
            fig.update_layout(title='Anomaly Identification', xaxis_title='Dimension-1', yaxis_title='Dimension-2')
            fig.show()

        elif dimensions == 3:
            fig=go.Figure(data=go.Scatter3d(x=outliers_sample['Dim-1'], y=outliers_sample['Dim-2'], z=outliers_sample['Dim-3'],
                                            mode='markers', name="Anomaly", marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=inliers_sample['Dim-1'], y=inliers_sample['Dim-2'], z=inliers_sample['Dim-3'],
                                     mode='markers', name="Normal",  marker=dict(size=5)))
            fig.update_layout(title='Anomaly Identification', xaxis_title='Dimension-1', 
                              yaxis_title='Dimension-2', zaxis_title='Dimension-3')
            fig.show()

        if return_data:
            return self.data
        


