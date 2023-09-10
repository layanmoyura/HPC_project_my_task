
# Load Data from Multiple Files with Balanced Load Distribution

## Running the Bash Script (Ubuntu)

To load data from multiple files where each file contains a different number of rows while maintaining a constant number of columns and ensuring a balanced load, you can use the provided Bash script. This script is designed to work on an Ubuntu operating system. In this program total rows loaded are 1M and it is divided into 4 files containing 100000, 200000, 300000, and 400000 respectively. 

To execute the script:

1. Open your terminal in the directory where your data files are located.

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

When dealing with multiple data files, it's common to have files with varying numbers of rows. This can be challenging when you want to distribute the data processing load evenly across available resources. The goal of this program is to load data from multiple files with different row counts while keeping the number of columns constant. By doing so, we aim to achieve a balanced load distribution for data processing tasks.

### Program Workflow

1. The Bash script (`part_2.sh`) first identifies all data files in the current directory.

2. It calculates the total number of rows in each file and stores this information.

3. The script determines the file with the maximum number of rows and uses this as the reference for load balancing.

4. It then calculates the number of rows each file should contribute to maintain a balanced load. This is done by dividing the maximum row count by the number of files.

5. The script then uses  relevant commands to extract the required number of rows from each file while preserving the constant number of columns.

6. The extracted data is processed as needed, and the balanced data is available for kmeans analysis.

7. Finally, the script may perform additional post-processing tasks or display the kmeans results.

### Customization

You can customize the script to meet your specific requirements, such as defining the number of columns or specifying the post-processing steps.

**Note**: Make sure you have the necessary data processing tools and dependencies installed to execute this script successfully.
