
# This code is importing necessary libraries and modules for the Python script.
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
from mpi4py import MPI


def plot(X, centroids, labels, show=True, iteration=None, file_name=None):
    """
    The function `plot` is used to visualize the K-means clustering algorithm by plotting the data
    points, centroids, and labels.
    
    Args:
      X: The parameter X is a numpy array representing the data points to be plotted. It should have
    shape (n, 2), where n is the number of data points and each data point has two features.
      centroids: The centroids parameter is a numpy array that represents the coordinates of the
    centroids in the plot. Each row of the array represents the coordinates of a centroid.
      labels: The "labels" parameter is a 1-dimensional array or list that assigns each data point in X
    to a specific cluster. Each element in the "labels" array corresponds to the cluster assignment of
    the corresponding data point in X.
      show: The "show" parameter determines whether the plot should be displayed immediately after it is
    created. If set to True, the plot will be shown. If set to False, the plot will not be displayed.
    Defaults to True
      iteration: The parameter "iteration" is used to specify the current iteration number of the
    K-means clustering algorithm. It is an optional parameter that can be used to provide additional
    information in the title of the plot.
      file_name: The `file_name` parameter is a string that specifies the name of the file to save the
    plot as. If provided, the plot will be saved as an image file with the specified name. If not
    provided, the plot will not be saved as a file.
    """
    
   # The below code is creating a scatter plot to visualize the results of the K-means clustering
   # algorithm. It plots the data points in X with different colors based on their assigned labels. It
   # also plots the centroids of each cluster as red crosses. The title, x-axis label, and y-axis
   # label are set accordingly. If the "show" parameter is set to True, the plot is displayed. If the
   # "iteration" and "file_name" parameters are provided, the plot is saved with the specified file
   # name.
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

# The `BaseModel` class is an abstract base class that defines the structure for fitting and
# predicting in machine learning models.
class BaseModel(ABC):
    
    def __init__(self) -> None:
        """
        The above function is a constructor that initializes an object.
        """
        super().__init__()
    
    @abstractmethod
    def fit(self, X, y):
        """
        The function "fit" is a placeholder that does not perform any operations.
        
        :param X: The X parameter represents the input data or features. It is a matrix or array-like
        structure where each row represents a sample and each column represents a feature or attribute
        of that sample. The shape of X is (n_samples, n_features), where n_samples is the number of
        samples and n_features is
        :param y: The parameter "y" represents the target variable or the dependent variable in the
        dataset. It is the variable that we want to predict or model using the features in the dataset
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        The function "predict" takes in a parameter "X" and does not perform any operations.
        
        :param X: The input data for which you want to make predictions
        """
        pass
    
# The `KMeans` class is a subclass of `BaseModel` that implements the K-means clustering algorithm.
class KMeans(BaseModel):
    
    def __init__(self, n_clusters, max_iter, comm, file_prefix) -> None: 
        """
        The function is a constructor that initializes the attributes of a clustering algorithm object.
        
        :param n_clusters: The number of clusters to be formed in the clustering algorithm
        :param max_iter: The maximum number of iterations for the clustering algorithm to run
        :param comm: The `comm` parameter is an object that represents the communication context. It is
        used for communication between different processes in parallel computing. It provides methods
        for sending and receiving data between processes. In this code, it is used to determine the rank
        and size of the current process. The rank represents the unique
        :param file_prefix: The `file_prefix` parameter is a string that represents the prefix of the
        file names that will be used for input and output files in the clustering algorithm. It is used
        to generate unique file names for each process in a parallel implementation of the algorithm
        """
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

        
        input_path = "kmeans_plots"
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        
    @property
    def labels(self):
        """
        The function returns the labels of an object.
        :return: The method is returning the value of the variable `self._labels`.
        """
        return self._labels
    
    @property
    def centroids(self):
        """
        The function returns the centroids of a given object.
        :return: The centroids of the object.
        """
        return self._centroids
    
    @property
    def initial_centroids(self):
        """
        The function returns the initial centroids.
        :return: The method is returning the value of the variable `_initial_centroids`.
        """
        return self._initial_centroids
    
    def _initialize_centroids(self, K: int, X: np.array) -> np.array:
        """
        The function initializes the centroids for a K-means clustering algorithm by randomly selecting
        K data points from the input array X.
        
        Args:
          K (int): The number of centroids to initialize.
          X (np.array): X is a numpy array containing the data points.
        
        Returns:
          the centroids, which is a numpy array.
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
        The function calculates the Euclidean distance between each point in X and each centroid in
        centroids.
        
        Args:
          centroids (np.array): The centroids parameter is a numpy array that represents the centroids
        of the clusters. Each row of the array represents a centroid, and the columns represent the
        features of the centroid.
          X (np.array): X is a numpy array representing the data points. Each row of X represents a data
        point, and each column represents a feature of that data point.
        
        Returns:
          an array of distances between each point in X and each centroid in centroids.
        """
       
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, distances: np.array) -> np.array:
        """
        The function assigns labels to data points based on the minimum distances.
        
        Args:
          distances (np.array): The distances parameter is a numpy array that represents the distances
        between data points and cluster centroids. Each row of the array corresponds to a data point,
        and each column corresponds to a cluster centroid. The value at position (i, j) in the array
        represents the distance between the i-th data point and
        
        Returns:
          the indices of the minimum values along the specified axis in the distances array.
        """
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.array, n_clusters: int, labels: np.array) -> np.array:
        """
        The function `_update_centroids` calculates the new centroids for each cluster by taking the mean
        of the data points assigned to that cluster and then performing an allreduce operation to
        synchronize the new centroids across all processes.
        
        Args:
          X (np.array): X is a numpy array representing the data points. Each row of X represents a data
        point, and each column represents a feature of that data point.
          n_clusters (int): The parameter `n_clusters` represents the number of clusters in the dataset.
        It is an integer value that specifies the desired number of clusters to be formed.
          labels (np.array): The `labels` parameter is a numpy array that contains the cluster
        assignments for each data point in the input data `X`. Each element in the `labels` array
        represents the cluster assignment for the corresponding data point in `X`.
        
        Returns:
          a numpy array of the updated centroids.
        """
        new_centroids = []
        for i in range(n_clusters):
            
            local_centroid = np.mean(X[labels == i], axis=0)
            start_com_time = time.time()            
            new_centroid_imd = self._comm.allreduce(local_centroid, op=MPI.SUM)
            end_com_time = time.time()
            elapsed_com_time = end_com_time - start_com_time
            self.spend_com_time += elapsed_com_time
            new_centroid = new_centroid_imd / self._size
            new_centroids.append(new_centroid)
        return np.array(new_centroids)     
        
    def fit(self, data_folder, DatasetSize, plot_graph, y=None) -> None:
        """
        The `fit` function is used to train a clustering model on a distributed dataset, with each
        process loading balanced portion of the data from multiple files and performing calculations on it.
        
        Args:
          data_folder: The `data_folder` parameter is the path to the folder where the data files are
        located. These data files should be in CSV format.
          DatasetSize: The parameter `DatasetSize` represents the total number of rows in the dataset
        that will be used for clustering. It is used to determine the number of rows that each process
        will load and process.
          plot_graph: The `plot_graph` parameter is a boolean flag that determines whether or not to
        plot the graph during the fitting process. If set to `True`, the graph will be plotted. If set
        to `False`, the graph will not be plotted. Defaults to False
          y: The parameter `y` is used to specify the target variable or labels for the data. It is
        optional and is not used in the `fit` method provided.
        """
        # The below code is implementing a parallel K-means clustering algorithm using MPI (Message
        # Passing Interface) in Python. It is divided into multiple processes, each responsible for
        # loading and processing a same size portion of the data. The code performs the following steps:
        start_read_time = time.time()

    
        data_files = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if filename.endswith(".csv")]
        
        rows_per_process = DatasetSize // self._size
        file_row_counts = [rows_per_process] * self._size
        file_row_counts = [ 100000, 200000, 300000, 400000] # Manually set the number of rows in each file

        start_row = 0

        data_parts = []
        rows_to_load = rows_per_process
        start_row = [0 , 150000, 200000,150000] # Manually set the start row for each process (change this such that each process loads the same number of rows)
        
    
        
        with open(data_files[self._rank], 'r') as file:

            data_chunk = np.genfromtxt(file, delimiter=',', skip_header=start_row[self._rank], max_rows=file_row_counts[self._rank])
            

            for row in data_chunk:
                try:
                    data_parts.append(row)
                    rows_to_load -= 1
                    if(rows_to_load == 0):
                        break
    
                except StopIteration:
                    break
            print(f"at process {self._rank} step 1: total rows loaded >>>> {len(data_parts)} and remaining rows to load >>>> {rows_to_load}")
                
        if self._rank != 3:  
            
            with open(data_files[self._rank + 1], 'r') as file:
                    
                data_chunk = np.genfromtxt(file, delimiter=',', skip_header=0, max_rows=start_row[self._rank + 1])
            

                for row in data_chunk:
                    
                    data_parts.append(row)
                    rows_to_load -= 1
                    if(rows_to_load == 0):
                        break
                print(f"at process {self._rank} step 2: total rows loaded >>>> {len(data_parts)} and remaining rows to load >>>> {rows_to_load}")
        
                    
        X = np.vstack(data_parts)
        
        end_read_time = time.time()
        elapsed_read_time = end_read_time - start_read_time
        self.spend_read_time = elapsed_read_time
        print(f"Process {self._rank}: Data loader took {elapsed_read_time:.4f} seconds")

        start_time = time.time()
        
        centroids = self._initialize_centroids(self._n_clusters, X)
        
        
        
        start_scatter_time = time.time()
        
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
        """
        The function predict takes in an array X and returns a numpy array, but it is not implemented
        yet.
        
        Args:
          X (np.array): An input array of shape (n_samples, n_features) where n_samples is the number of
        samples and n_features is the number of features.
        
        Returns:
          The code is returning a `NotImplemented` error message.
        """
        return NotImplemented("Not implemented")


