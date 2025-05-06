import pandas as pd


def load_data(file_path: str, useless_cols = ["original_listed_time","listed_time","job_posting_url","application_url","expiry", "closed_time", "posting_domain","sponsored","compensation_type", "formatted_work_type"]) -> pd.DataFrame:
    """
    Loads the data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Check if the file path is empty
    if not file_path:
        raise ValueError("The file path cannot be empty.")

    # Load the data
    df = pd.read_csv(file_path, index_col=0)

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The loaded DataFrame is empty.")

    # Drop useless columns
    if useless_cols:
        df.drop(columns=useless_cols, inplace=True)

    #lowercase all text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].str.lower()

    return df



def transform_data(df: pd.DataFrame, predict: str = "salary") -> (pd.DataFrame, pd.Series):
    """
    Transforms the input DataFrame to be ready for ML modeling.

    Args:
        df (pd.DataFrame): The input DataFrame to transform.
        regress_on (str): [either 'salary' or 'views'] The target variable to predict.

    Returns:
        pd.DataFrame, pd.Series: The transformed DataFrame and the target variable.
    """
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if predict not in ["salary", "views"]:
        raise ValueError("The 'predict' argument must be either 'salary' or 'views'.")

    #features = ["company_name", "title", "description"] if features is None else features  # we'll see if we need other features ,"location","work_type", "zip_code", "fips"]
    
    if predict == "salary":
        if "normalized_salary" not in df.columns:
            raise ValueError("The DataFrame does not contain a 'normalized_salary' column.")
        df.dropna(subset=["normalized_salary"], inplace=True)
        y = df["normalized_salary"]
        df.drop(columns=["normalized_salary"], inplace=True)
    elif predict == "views":
        if "views" not in df.columns:
            raise ValueError("The DataFrame does not contain a 'views' column.")

        df.dropna(subset=["views"], inplace=True)
        y = df["views"]
        df.drop(columns=["views"], inplace=True)

    # Drop useless columns


    return df, y
