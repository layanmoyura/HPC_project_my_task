# The lines `import os` and `import numpy as np` are importing two Python libraries: `os` and `numpy`.
import os
import numpy as np

def generateData(output_folder='data_files'):
    """
    The function `generateData` generates random data and saves it as CSV files in the specified output
    folder.
    
    Args:
      output_folder: The output_folder parameter is a string that specifies the folder where the
    generated data files will be saved. By default, it is set to 'data_files'. Defaults to data_files
    """
    
    np.random.seed(123)

    
    file_sizes = [100000, 200000, 300000, 400000]

    
    os.makedirs(output_folder, exist_ok=True)

    for i, size in enumerate(file_sizes):
       
        X = np.random.rand(size, 2)

       
        filename = os.path.join(output_folder, f'data_{i + 1}.csv')
        np.savetxt(filename, X, delimiter=',')


# The line `generateData(output_folder='data_folder')` is calling the `generateData` function with the
# argument `output_folder='data_folder'`. This means that the function will generate random data and
# save it as CSV files in the folder named 'data_folder'.
generateData(output_folder='data_folder')
