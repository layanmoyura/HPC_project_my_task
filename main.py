# These lines of code are importing necessary modules and classes for the program.
from mpi4py import MPI
import csv
import os
from kmeans import KMeans  

# The code is initializing variables `data_folder`, `K`, `max_iter`, and `row_sizes`.
data_folder = "data_folder" 
K = 3  
max_iter = 31  
row_sizes = []  

# This code block is iterating over the files in the `data_folder` directory and checking if each file
# has a ".csv" extension. If a file has a ".csv" extension, it opens the file, reads its contents
# using the `csv.reader` function, and stores the number of rows in the `row_sizes` list along with
# the filename.
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder, filename)


        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            num_rows = len(rows)
            row_sizes.append((filename, num_rows))  

total_size = sum([row_size[1] for row_size in row_sizes])
print(f"Total number of rows: {total_size}")

# The code `comm = MPI.COMM_WORLD` creates a communicator object `comm` that represents the group of
# processes involved in the parallel computation.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# The code `kmeans = KMeans(n_clusters=K, max_iter=max_iter, comm=comm,
# file_prefix="kmeans_plots/kmeans_clustering")` creates an instance of the `KMeans` class with the
# specified parameters. The `KMeans` class is a custom class that implements the k-means clustering
# algorithm.
kmeans = KMeans(n_clusters=K, max_iter=max_iter, comm=comm, file_prefix="kmeans_plots/kmeans_clustering")
kmeans.fit(data_folder, total_size, True)  

# The code block you provided is generating a GIF animation from a series of images.
from util import generate_gif

if rank == 0:
    
    input_path = "kmeans_plots"
    file_prefix = "kmeans_clustering"
    output_path = "images"
    output_file = "kmeans_clustering_animate.gif"
    duration = 1000  

    try:
        generate_gif(path=input_path, file_prefix=file_prefix, output_path=output_path, output_file=output_file, duration=duration) 
    except Exception as e:
        print({"Error": str(e)})
        print("Error: Could not generate GIF animation")