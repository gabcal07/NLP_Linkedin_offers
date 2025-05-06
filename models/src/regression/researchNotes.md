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