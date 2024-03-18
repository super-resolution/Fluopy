import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class ExperimentalData:
    """
    Container of experimental-data-associated attributes and methods.
    """
    def __init__(self, file):
        """
        Parameters
        ----------
        file : str
            The path of the .txt file.
        """
        self.data = pd.read_csv(file, sep=' ', skiprows=1,
                                names=['x_position', 'x_uncertain', 'y_position', 'y_uncertain', 'frame', 'intensity', 'fit', 'background'])

    def cluster(self, eps=20, min_samples=3):
        """
        Cluster the 2-D data with DBSCAN.

        Parameters
        ----------
        eps : int
            The maximum distance between two samples for one to be considered as in
            the neighborhood of the other.
        min_samples : int
            The number of samples in a neighborhood for a point to be considered a core
            point. See sklearn.cluster.DBSCAN for more information.
        """
        coordinates = np.vstack((self.data['x_position'], self.data['y_position'])).T
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

        self.data['cluster_id'] = db.labels_

    def get_event_time_series(self, index, frame_time):
        """
        Counts events within a time interval (frame_time).

        Parameters
        ----------
        index : int
            Cluster id (label) of which event_time_series is constructed.
        frame_time : str
            For possible input values, see https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.

        Returns
        -------
        event_time_series : pd.Series
            Contains the time points as index and the number of events as values.
        """
        frame_time = pd.Timedelta(frame_time) / np.timedelta64(1, 's')
        time_stamps = np.linspace(0, frame_time*self.data['frame'].max(), self.data['frame'].max())
        current_data = self.data[self.data['cluster_id'] == index]
        event_time_series = pd.Series(np.zeros(self.data['frame'].max()), index=time_stamps)
        event_time_series.values[current_data['frame']] = current_data['intensity']
        
        return event_time_series

    def plot_cluster(self):
        """
        Plot the clusters.
        """
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
        