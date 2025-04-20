import kagglehub
import os 
# Download latest version to the specified directory
path = kagglehub.dataset_download("arshkon/linkedin-job-postings")

print(f"Path to dataset files: {path}")
print(f"List of files in the dataset: {os.listdir(path)}")