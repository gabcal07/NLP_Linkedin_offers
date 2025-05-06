from metrics import (
    grammar_metric,
    long_range_coherence_metric,
    novelty_metric,
    semantic_richness_metric,
    professionalism_metric,
)
import mlflow
import pandas as pd
import dotenv

dotenv.load_dotenv()

def evaluate_dataset(dataframe_path: str) -> None:
    """
    Evaluate the dataset statically using various metrics.
    """

    dataframe = pd.read_csv(dataframe_path)
    # ensure that dataframe contains 'predictions' and 'ground_truth' columns
    if (
        "predictions" not in dataframe.columns
        or "ground_truth" not in dataframe.columns
    ):
        raise ValueError(
            "DataFrame must contain 'predictions' and 'ground_truth' columns."
        )
    # Evaluate the dataset using the defined metrics
    extra_metrics = [
        grammar_metric,
        long_range_coherence_metric,
        novelty_metric,
        semantic_richness_metric,
        professionalism_metric,
        mlflow.metrics.rougL(),
        mlflow.metrics.bleu(),
        mlflow.metrics.flesch_kincaid_grade_level(),
    ]
    # Evaluate the dataset using the defined metrics
    with mlflow.start_run():
        results = mlflow.evaluate(
            model = None,
            data=dataframe,
            predictions="predictions",
            ground_truth="ground_truth",
            extra_metrics=extra_metrics,
    )
    print(results.metrics)
    print(results.tables["eval_results_table"])