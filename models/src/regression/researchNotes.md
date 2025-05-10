## did some statistical analysis to determine what columns are useless for our use cases:
✅ : validated choice

⚠️ : up to interpretation

❌ : not validated choice


### postings.csv:

* ✅ `compensation_type`:  only one unique value for the entire dataset: `"BASE_SALARY"`=> **this column does not provide any useful meaning for it to be used in ML treatements**


* ✅ `sponsored`: only one unique value for the entire dataset: `0` => **this column does not provide any useful meaning for it to be used in ML treatements**


* ⚠️ `allowed_remote`: only null and 1 values: we cannot assume null values are 0, so we have to drop this column as well => **this column does not provide any useful meaning for it to be used in ML treatements**


* ⚠️ `application_type`: might have impact on salary (statement drawn from quick boxplot analysis) but not sure if it is worth keeping it.

## Statistical 


## Unformated notes

table result with following features:
* categorical features: `"work_type"`, `"formatted_experience_level"`, `"pay_period`, `"state"`(engineered from `"location"`), `"company_id"`
* text features: `"title"`, `"description"`
* numerical features: `avg_salary` (engineered:average salary for a given company), `median_salary` (engineered:median salary for a given company), `postings_count` (engineered: number of postings for a given company)

| Model                                                                            | Train MAE | Train RMSE | Train R² | Test MAE  | Test RMSE | Test R² | CV MAE (k=5) | CV RMSE (k=5) | CV R² (k=5) |
|----------------------------------------------------------------------------------|-----------|------------|----------|-----------|-----------|---------|--------------|---------------|-------------|
| RandomForest(n_estimators=30) + TfIDF(max_features=100,n_gram=(1,1))             | $5 794.2  | $9 537.4   | 0.97     | $14 583.5 | $23 781.4 | 0.78    | $16 174.4    | $25 673.2     | 0.749       |
| RandomForest(max_depth=8 n_estimators=50) + TfIDF(max_features=100,n_gram=(1,1)) | $16 963.9 | $25503.7   | 0.753    | $17 736.8 | $26 891.9 | 0.721   | 0.567        | 0.890         | 0.705       |
| RandomForest(max_depth=8 n_estimators=50) + TfIDF(max_features=100,n_gram=(1,2)) | 0.234     | 0.567      | 0.890    | 0.345     | 0.678     | 0.901   | 0.456        | 0.789         | 0.123       |