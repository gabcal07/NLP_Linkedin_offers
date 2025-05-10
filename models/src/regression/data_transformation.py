import os

import kagglehub
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame


def clean_data(df:pd.DataFrame, target:str="normalized_salary") -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing outliers and filling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        target (str): The target variable to use for outlier removal.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Check if the DataFrame is empty
    allowed_targets = ["normalized_salary", "views", "applies"]

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    # Check if the target column exists
    if target not in df.columns:
        raise ValueError(f"The DataFrame does not contain a '{target}' column.")
    if target not in allowed_targets:
        raise ValueError(f"The target must be one of {allowed_targets}.")


    # Check if the target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"The '{target}' column must be numeric.")


    # Drop useless columns:
    useless_cols = [
        "max_salary",
        "med_salary",
        "min_salary",
        "formatted_work_type",
        "original_listed_time",
        "remote_allowed",
        "job_posting_url",
        "application_url",
        "expiry",
        "closed_time",
        "listed_time",
        "posting_domain",
        "sponsored",
        "compensation_type",
        "zip_code",
        "fips",
        "currency",
        #"pay_period" # not sure about this one
        "application_type"
    ]

    df = df.drop(columns=useless_cols, errors='ignore')

    df = df.dropna(subset=["company_name", "title", "description", "company_id"])

    # fill nan values
    df.loc[:,"pay_period"] = df["pay_period"].fillna("UNKNOWN")
    df.loc[:,"formatted_experience_level"] = df["formatted_experience_level"].fillna("Not specified")
    df.loc[:, "skills_desc"] = df["skills_desc"].fillna("Not specified")
    df.loc[:,"applies"] = df["applies"].fillna(0)
    df.loc[:,"views"] = df["views"].fillna(0)

    if target == "normalized_salary":
        df = df.dropna(subset=["normalized_salary"])
        df = df.drop(columns=["views", "applies"], errors='ignore')

    elif target == "views":
        df = df.dropna(subset=["views"])
        df = df.drop(columns=["normalized_salary"], errors='ignore')

    elif target == "applies":
        df = df.dropna(subset=["applies"])
        df = df.drop(columns=["normalized_salary"], errors='ignore')


    # Remove outliers
    lower_bound = df[target].quantile(0.01)
    upper_bound = df[target].quantile(0.99)
    df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]


    return df




def load_data(target="normalized_salary"):
    path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
    postings_raw = pd.read_csv(os.path.join(path, "postings.csv"), index_col=0)
    df = clean_data(postings_raw, target=target)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

