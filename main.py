from mpi4py import MPI
from kmeans import KMeans  # Import your KMeans class module here

# Load data from CSV files
data_folder = "data_files"  # Specify the folder containing the data files

# Define parameters for KMeans
K = 3  # Number of clusters
max_iter = 3  # Maximum number of iterations

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a KMeans instance and fit it to the data
kmeans = KMeans(n_clusters=K, max_iter=max_iter, comm=comm, file_prefix="kmeans_plots/kmeans_clustering")
kmeans.fit(data_folder, 1000000, False)  # Assuming a total data size of 1,000,000 rows


# Generate a GIF animation of the clustering process
from util import generate_gif

if rank == 0:
    # Define input and output paths and file prefix
    input_path = "kmeans_plots"
    file_prefix = "kmeans_clustering"
    output_path = "images"
    output_file = "kmeans_clustering_animate.gif"
    duration = 1000  # Duration between frames in milliseconds

    # Generate the GIF animation
    try:
        generate_gif(path=input_path, file_prefix=file_prefix, output_path=output_path, output_file=output_file, duration=duration) 
    except Exception as e:
        print({"Error": str(e)})
        print("Error: Could not generate GIF animation")