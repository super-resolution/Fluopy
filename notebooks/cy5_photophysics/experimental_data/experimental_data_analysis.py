import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class ExperimentalData:
    def __init__(self, file):
        self.data = pd.read_csv(file, sep=' ', skiprows=1,
                                names=['x_position', 'x_uncertain', 'y_position', 'y_uncertain', 'frame', 'intensity', 'fit', 'background'])
        
    def cluster(self, eps=20, min_samples=3):
        coordinates = np.vstack((self.data['x_position'], self.data['y_position'])).T
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

        self.data['cluster_id'] = db.labels_

    def plot(self):
        if not 'cluster_id' in self.data.columns:
            raise ValueError('data has to be clustered first.')
        unique_labels = self.data['cluster_id'].unique()
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, unique_labels.size)]
        
        groups = self.data.groupby('cluster_id')
        for (cluster_id, group), color in zip(groups, colors):
            if cluster_id == -1:
                color = [0, 0, 0, 1]
            
            plt.plot(group['x_position'], group['y_position'], 
                    "o", ms=0.1, color=color)
            



        