import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline import TextTransformer, PretrainedWord2VecVectorizer, build_pipeline, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_transformation import clean_data, load_data

import kagglehub
import os
import pandas as pd
import json


word2vec_models = ["glove-wiki-gigaword-50", "word2vec-google-news-300"]#, "glove-wiki-gigaword-200", "glove-wiki-gigaword-300", "word2vec-google-news-300", "fasttext-wiki-news-subwords-300"]
tfidf_params = [

    {
        "max_features": 1000,
        "ngram_range": (1, 1),
    },
    {
        "max_features": 1000,
        "ngram_range": (1, 2),
    }
]

categorical_variables = ["pay_period", "formatted_experience_level", "work_type"]

text_variables_subsets = [
    ["title"],
    ["title", "company_name"],
    ["title", "description", "company_name","skills_desc", "location"]
]




def evaluate_linear_regression(X, y, test_size=0.2, cv=5):

    metrics = {}
    print("Evaluating linear regression models...")

    for params in tfidf_params:
        print(f"Evaluating model with tfidf params: {params}")
        for text_cols in text_variables_subsets:
                if not text_cols:
                    continue
                print(f"Evaluating model with tfidf params: {params}, text columns: {text_cols}, one-hot columns: {categorical_variables}")
                pipeline = build_pipeline(one_hot_cols=categorical_variables,
                                          text_cols=[(col, TfidfVectorizer(**params)) for col in text_cols],
                                          model=LinearRegression()
                                          )
                res = evaluate_model(pipeline, X, y, test_size=test_size, cv=cv)
                metrics[(str(params), tuple(text_cols))] = res

    for word2vec_model in word2vec_models:
        print(f"Evaluating model with word2vec model: {word2vec_model}")
        for text_cols in text_variables_subsets:
                if not text_cols:
                    continue
                print(f"Evaluating model with word2vec model: {word2vec_model}, text columns: {text_cols}, one-hot columns: {categorical_variables}")
                pipeline = build_pipeline(one_hot_cols=categorical_variables,
                                          text_cols=[(col, PretrainedWord2VecVectorizer(word2vec_model)) for col in text_cols],
                                          model=LinearRegression()
                                          )
                res = evaluate_model(pipeline, X, y, test_size=test_size, cv=cv)


                metrics[(word2vec_model, tuple(text_cols))] = res



    return metrics



if __name__ == "__main__":
    X,y = load_data()
    metrics = evaluate_linear_regression(X, y)
    # Save the metrics to a json file
    with open("metrics_linear_regrssion.json", "w") as f:
        json.dump(metrics, f, indent=4)





