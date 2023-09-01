import os
import numpy as np

def generateData(output_folder='data_files'):
    # Set random seed for reproducibility
    np.random.seed(123)

    # Define the desired sizes for each file
    file_sizes = [100000, 200000, 300000, 400000]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i, size in enumerate(file_sizes):
        # Generate random data
        X = np.random.rand(size, 2)

        # Save data to a unique CSV file in the output folder
        filename = os.path.join(output_folder, f'data_{i + 1}.csv')
        np.savetxt(filename, X, delimiter=',')

# Call the function to generate four files in the "data_files" folder
generateData(output_folder='data_files')
