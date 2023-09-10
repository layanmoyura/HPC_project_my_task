
# Load Data from Multiple Files with Balanced Load Distribution

## Running the Bash Script (Ubuntu)

To load data from multiple files where each file contains a different number of rows while maintaining a constant number of columns and ensuring a balanced load, you can use the provided Bash script. This script is designed to work on an Ubuntu operating system. In this program total rows loaded are 1M and it is divided into 4 files containing 100000, 200000, 300000, and 400000 respectively as shown in below figure. Each process load 250000 rows of data from four files. 

![1](https://github.com/layanmoyura/HPC_project_my_task/assets/84334230/0de2d746-2d56-44e1-a9c2-b247e243657f)


To execute the script:

1. Open your terminal in the directory where your program files are located.

2. Make sure the script has execute permissions. If not, you can grant execute permissions by running:

   ```bash
   chmod +x part_2.sh
   ```

3. Run the script:

   ```bash
   ./part_2.sh
   ```

## Program Description

### Problem Statement

When dealing with multiple data files, it's common to have files with varying numbers of rows. This can be challenging when you want to distribute the data processing load evenly across available resources. The goal of this program is to load data from multiple files with different row counts while keeping the number of columns constant. By doing so, we aim to achieve a balanced load distribution for data processing tasks. Whenever the file size or file structure is chaneged the starting cursor points should be manually set in the program.

### Program Workflow

1. The Bash script (`part_2.sh`) first identifies all data files in the current directory.

2. It calculates the total number of rows in each file and stores this information.

3. The script determines the file with the maximum number of rows and uses this as the reference for load balancing.

4. It then calculates the number of rows each file should contribute to maintain a balanced load. This is done by dividing the maximum row count by the number of files.

5. The script then uses  relevant commands to extract the required number of rows from each file while preserving the constant number of columns.

6. The extracted data is processed as needed, and the balanced data is available for kmeans analysis.

7. Finally, the script may perform additional post-processing tasks or display the kmeans results.

### Results Displayed in the Console:

The script provides various information in the console during its execution. Here's a list of results and information that can be shown:

1. Process Information: Each process may display its rank and step information. For example, "at process 0 step 1" and "at process 0 step 2" indicate loading data in two steps for process 0.

2. Data Loading Time: The script measures and displays the time taken for loading data from CSV files, e.g., "Process 0: Data loader took 2.3456 seconds."

3. Calculation Time: It displays the time taken for the actual K-means clustering calculations, e.g., "Process 0: Calculation took 1.2345 seconds."

4. Communication Time: The time spent on communication between processes using MPI is also displayed, e.g., "Process 0: Communication took 0.5678 seconds."

5. Total Time Breakdown: At the end of the script, the total time spent on data loading, communication, and calculation is summarized, e.g., "Data loader took 3.1234 seconds," "Communication took 0.7890 seconds," and "Calculation took 2.3456 seconds."

6. Finally Kmeans iterations as shown below saved in images folder.

![kmeans_clustering_animate](https://github.com/layanmoyura/HPC_project_my_task/assets/84334230/233e4d30-af5d-4d4f-8abf-ebc21609198c)


### Customization

You can customize the script to meet your specific requirements, such as defining the number of columns or specifying the post-processing steps.

**Note**: Make sure you have the necessary data processing tools and dependencies installed to execute this script successfully.
