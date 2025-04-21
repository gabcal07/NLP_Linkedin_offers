import kagglehub
import os
import pandas as pd
import re

# Download latest version to the specified directory
path = kagglehub.dataset_download("arshkon/linkedin-job-postings")

# This function cleans the text by:
# 1. Converting to lowercase
# 2. Removing URLs
# 3. Removing email addresses
# 4. Removing numbers with 10 or more digits
# 5. Removing special characters
# 6. Removing extra whitespace
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d{10,}", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    print(f"Path to dataset files: {path}")
    print(f"List of files in the dataset: {os.listdir(path)}")

    postings_path = path + "/postings.csv"
    postings_df = pd.read_csv(
        postings_path, usecols=["title", "location", "company_name", "description"]
    )

    rows_with_any_nan = (
        postings_df[["company_name", "description", "title"]].isna().any(axis=1).sum()
    )
    print(f"Rows with at least one NaN value: {rows_with_any_nan}")

    # drop rows with NaN values in specific columns
    print(f"Number of rows before dropping NaN values: {postings_df.shape[0]}")
    postings_df.dropna(subset=["company_name", "description", "title"], inplace=True)

    print(f"Number of rows after dropping NaN values: {postings_df.shape[0]}")
    postings_df.reset_index(drop=True, inplace=True)

    # Apply the clean_text function and ASSIGN the result back
    postings_df["description"] = postings_df["description"].apply(
        lambda x: clean_text(x)
    )

    # Create the processed directory if it doesn't exist
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Save the DataFrame to the processed directory
    output_path = os.path.join(processed_dir, "cleaned_postings_modeling.parquet")
    postings_df.to_parquet(output_path)
    print(f"DataFrame saved to: {output_path}")
