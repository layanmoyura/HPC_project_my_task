import os
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
from mpi4py import MPI


def plot(X, centroids, labels, show=True, iteration=None, file_name=None):
    # Plot the original data and clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title('K-means Clustering iteration ' + str(iteration))
    plt.xlabel('X')
    plt.ylabel('Y')
    if show:
        plt.show()
    if iteration and file_name:
        file_name = file_name + "_" + str(iteration)
        plt.savefig(file_name)
    plt.close()

class BaseModel(ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
class KMeans(BaseModel):
    
    def __init__(self, n_clusters, max_iter, comm, file_prefix) -> None: 
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._centroids = None
        self._labels = None
        self._file_prefix = file_prefix
        self._initial_centroids = None
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self.spend_time = 0
        self.spend_com_time = 0
        self.spend_read_time = 0

        # Create the kmeans_plots folder if it doesn't exist
        input_path = "kmeans_plots"
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def centroids(self):
        return self._centroids
    
    @property
    def initial_centroids(self):
        return self._initial_centroids
    
    def _initialize_centroids(self, K: int, X: np.array) -> np.array:
        """
        Calculates the initial centroids

        Args:
            K (int): number of clusters
            X (np.array): training data

        Returns:
            np.array: initial centroids
        """
        centroids = None
        if self._rank == 0:
            centroid_indices = np.random.choice(len(X), K, replace=False)
            centroids = X[centroid_indices.tolist()]
        centroids = self._comm.bcast(centroids, root=0)
        self._initial_centroids = centroids
        return centroids
    
    def _calculate_euclidean_distance(self, centroids: np.array, X: np.array) -> np.array:
        """
        Calculates the Euclidean distance of centroids and data
        
        Args:
            centroids (np.array): cluster centroids
            X (np.array): data

        Returns:
            np.array: distance as an array
        """
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, distances: np.array) -> np.array:
        """
        Assign labels for data points

        Args:
            distances (np.array): Euclidean distances

        Returns:
            np.array: labels as an integer array
        """
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.array, n_clusters: int, labels: np.array) -> np.array:
        """
        Update centroids
        Args:
            X (np.array): data
            n_clusters (int): number of clusters
            labels (np.array): cluster id per each data point

        Returns:
            np.array: updated centroids
        """
        new_centroids = []
        for i in range(n_clusters):
            # Local mean of the centroid belonging to cluster `i`
            local_centroid = np.mean(X[labels == i], axis=0)
            start_com_time = time.time()
            # Global sum of local mean of the centroid belonging to cluster `i`
            new_centroid_imd = self._comm.allreduce(local_centroid, op=MPI.SUM)
            end_com_time = time.time()
            elapsed_com_time = end_com_time - start_com_time
            self.spend_com_time += elapsed_com_time
            # Mean value of the global sum taken by dividing with the number of total processes
            new_centroid = new_centroid_imd / self._size
            new_centroids.append(new_centroid)
        return np.array(new_centroids)     
        
    def fit(self, data_folder, DatasetSize, plot_graph=False, y=None) -> None:
        """
        Training the KMeans algorithm

        Args:
            data_folder (String): folder containing data files
            y : Ignored but placed as a convention.
        """
        # Load data from multiple files and balance it among processes
        start_read_time = time.time()

    # Calculate how many data points each process should load
        data_files = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if filename.endswith(".csv")]

        # Determine the total number of rows in each file
        #file_row_counts = []
        #for data_file in data_files:
            #with open(data_file, 'r') as file:
               # num_rows = sum(1 for line in file)
                #file_row_counts.append(num_rows)

        # Calculate the total number of rows to be loaded across all processes
        total_rows = DatasetSize
        rows_per_process = total_rows // self._size
        file_row_counts = [ 100000, 200000, 300000, 400000]

        # Calculate the starting row for each process
        start_row = 0
        #for i in range(self._rank):
           #start_row += file_row_counts[i]
        
        # Calculate the number of rows to load for this process
        #rows_to_load = min(rows_per_process, total_rows - start_row)
        

        # Load data from the files in chunks
        data_parts = []
        temp_row = 0
        temp_index = 0
        for data_file in data_files:
            
            if temp_index == 1:
                rows_to_load = temp_row
                temp_index = 0
            else:
                rows_to_load = rows_per_process
            
            with open(data_file, 'r') as file:
                                        
                    while rows_to_load > 0:
                        try:
                            line = next(file).strip()
                            data_part = np.fromstring(line, sep=',')
                            data_parts.append(data_part)
                            rows_to_load -= 1
    
                        except StopIteration:
                            temp_row = rows_to_load
                            temp_index = 1
                            break
                        
                        
                
        
        X = np.vstack(data_parts)
        print(X.size)
        
        end_read_time = time.time()
        elapsed_read_time = end_read_time - start_read_time
        self.spend_read_time = elapsed_read_time
        print(f"Process {self._rank}: Data loader took {elapsed_read_time:.4f} seconds")

        start_time = time.time()
        # Initialize centroids
        centroids = self._initialize_centroids(self._n_clusters, X)
        
        
        
        start_scatter_time = time.time()
        # Scatter data
        x_local = np.empty((X.shape[0] // self._size, X.shape[1]), dtype=X.dtype)
        self._comm.Scatter(X, x_local, root=0)
        end_scatter_time = time.time()
        
        if self._rank == 0:
            elapsed_scatter_time = end_scatter_time - start_scatter_time
            self.spend_com_time += elapsed_scatter_time
        labels = None
        for i in range(self._max_iter):
            distances = self._calculate_euclidean_distance(centroids, x_local)

            labels = self._assign_labels(distances)
        
            centroids = self._update_centroids(x_local, self._n_clusters, labels)
            
            # If file_prefix is provided, create plots at each iteration
            if plot_graph and self._rank == 0 and self._file_prefix:
                plot(x_local, centroids, labels, False, i, self._file_prefix)
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.spend_time = elapsed_time

        print(f"Process {self._rank}: Calculation took {self.spend_time - self.spend_com_time:.4f} seconds")
        print(f"Process {self._rank}: Communication took {self.spend_com_time:.4f} seconds")

        self._centroids = centroids
        self._labels = labels

        final_spend_read_time = self._comm.allreduce(self.spend_read_time, op=MPI.MIN)
        final_spend_com_time = self._comm.allreduce(self.spend_com_time, op=MPI.MAX)
        final_spend_calc_time = self._comm.allreduce(self.spend_time - self.spend_com_time, op=MPI.MAX)

        if self._rank == 0:
            print(f"\033[1;32mData loader took {final_spend_read_time:.4f} seconds\033[0m")
            print(f"\033[1;33mCommunication took {final_spend_com_time:.4f} seconds\033[0m")
            print(f"\033[1;31mCalculation took {final_spend_calc_time:.4f} seconds\033[0m")

    
    def predict(self, X: np.array) -> np.array:
        return NotImplemented("Not implemented")


