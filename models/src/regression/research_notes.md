# Regression

## Models:

### :

MSE: 11745.31

|                            Model                            |   MSE    |                  Predictions                  |
| :---------------------------------------------------------: | :------: | :-------------------------------------------: |
|       TF IDF (max feat. 1000) + Box Cox + Linear Reg        | 11745.31 | Float nb / bad predictions /train time ~ 1min |
| TF IDF (max feat. 1000) + Standard Scaler + Gamma Regressor | 11664.86 |          Float nb / bad predictions           |

### Notes:

Nb features for TF IDF + Standard Scaler + Gamma Regressor doesnt change much

# Classification

## Models:

TF-IDF (feat=2500 / ng=(1,2) / min=2 / max=0.8) + MultinomialNB:

|    class     | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | :------: | ------- |
|     ACCT     |   0.460   | 0.546  |  0.499   | 1017    |
|     ADM      |   0.436   | 0.327  |  0.374   | 938     |
|     ADVR     |   0.409   | 0.223  |  0.289   | 121     |
|     ANLS     |   0.087   | 0.003  |  0.005   | 767     |
|     ...      |    ...    |  ...   |   ...    | ...     |
|   accuracy   |    ...    |  ...   |  0.381   | 41154   |
|  macro_avg   |   0.328   | 0.244  |  0.248   | 41154   |
| weighted_avg |   0.364   | 0.381  |  0.348   | 41154   |

---

TF-IDF (feat=5000 / ng=(1,2) / min=2 / max=0.8) + MultinomialNB:

|    class     | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | :------: | ------- |
|     ACCT     |   0.463   | 0.599  |  0.522   | 1017    |
|     ADM      |   0.418   | 0.353  |  0.383   | 938     |
|     ADVR     |   0.409   | 0.223  |  0.289   | 121     |
|     ANLS     |   0.174   | 0.016  |  0.029   | 767     |
|     ...      |    ...    |  ...   |   ...    | ...     |
|   accuracy   |    ...    |  ...   |  0.387   | 41154   |
|  macro avg   |   0.328   | 0.264  |  0.266   | 41154   |
| weighted avg |   0.369   | 0.387  |  0.361   | 41154   |
