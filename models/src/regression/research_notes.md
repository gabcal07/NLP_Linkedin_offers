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

---

TF-IDF (feat=5000 / ng=(1,2) / min=2 / max=0.8) + MultinomialNB:

|    class     | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | :------: | ------- |
|     ...      |    ...    |  ...   |   ...    | ...     |
|   accuracy   |           |        |  0.389   | 41154   |
|  macro avg   |   0.342   | 0.262  |  0.263   | 41154   |
| weighted avg |   0.376   | 0.389  |  0.364   | 41154   |

---

### GridSearchCV for optimal MultinomialNB:

Best Parameters: {'nb**nb**alpha': 0.1, 'text_preprocessing**tfidf**max_df': 1.0, 'text_preprocessing**tfidf**max_features': 10000, 'text_preprocessing**tfidf**min_df': 10, 'text_preprocessing**tfidf**ngram_range': (1, 2)}
Best F1-Weighted Score: 0.376014236224394
precision recall f1-score support

        ACCT      0.442     0.612     0.513      1017
         ADM      0.419     0.391     0.405       938
        ADVR      0.382     0.347     0.364       121
        ANLS      0.195     0.040     0.067       767
         ART      0.192     0.101     0.132       288
          BD      0.422     0.233     0.300      2739
        CNSL      0.230     0.107     0.146       428
        CUST      0.411     0.196     0.266       816
        DIST      0.224     0.278     0.249        79
        DSGN      0.259     0.113     0.157       452
         EDU      0.365     0.580     0.448       426
         ENG      0.387     0.309     0.344      2470
         FIN      0.304     0.388     0.341      1601
        GENB      0.422     0.262     0.324       381
        HCPR      0.712     0.897     0.794      3308
          HR      0.596     0.520     0.555       496
          IT      0.420     0.523     0.466      5079
         LGL      0.697     0.696     0.696       496
        MGMT      0.272     0.214     0.239      4122
        MNFC      0.329     0.454     0.382      3549
        MRKT      0.280     0.412     0.334      1043
        OTHR      0.301     0.229     0.260      2417
          PR      0.160     0.087     0.113       263
        PRCH      0.162     0.132     0.145       129
        PRDM      0.233     0.385     0.291       270
        PRJM      0.258     0.413     0.318       734
        PROD      0.153     0.081     0.106       136
          QA      0.282     0.318     0.299       349
        RSCH      0.275     0.338     0.303       577
        SALE      0.361     0.363     0.362      4322
         SCI      0.211     0.172     0.189       157
        STRA      0.108     0.052     0.070       232
        SUPL      0.162     0.215     0.185       228
        TRNG      0.223     0.048     0.079       440
         WRT      0.163     0.106     0.128       284

    accuracy                          0.387     41154

macro avg 0.315 0.303 0.296 41154
weighted avg 0.374 0.387 0.370 41154

### With adding "title" column

              precision    recall  f1-score   support

        ACCT      0.482     0.694     0.569      1017
         ADM      0.496     0.482     0.489       938
        ADVR      0.323     0.430     0.369       121
        ANLS      0.233     0.154     0.185       767
         ART      0.242     0.278     0.259       288
          BD      0.410     0.419     0.415      2739
        CNSL      0.220     0.187     0.202       428
        CUST      0.362     0.261     0.303       816
        DIST      0.253     0.278     0.265        79
        DSGN      0.257     0.195     0.221       452
         EDU      0.345     0.556     0.426       426
         ENG      0.401     0.480     0.437      2470
         FIN      0.347     0.453     0.393      1601
        GENB      0.410     0.270     0.326       381
        HCPR      0.784     0.914     0.844      3308
          HR      0.570     0.746     0.646       496
          IT      0.508     0.399     0.447      5079
         LGL      0.675     0.774     0.721       496
        MGMT      0.328     0.178     0.230      4122
        MNFC      0.369     0.507     0.427      3549
        MRKT      0.321     0.424     0.365      1043
        OTHR      0.473     0.378     0.420      2417
          PR      0.156     0.152     0.154       263

...
accuracy 0.420 41154
macro avg 0.335 0.377 0.346 41154
weighted avg 0.417 0.420 0.408 41154

---

### Notes on GridSearchCV:

The optimal model is not that good but an issue is that there are a lot of classes and a lot of confusion between the classes, e.g for "BD" class 43% percent of predicitons are "SALE".
This means that the Naive Bayes model struggles to differentiate between the classes.

#### Adding the "Title" column:

We have a noticeably better score with the "title" column added to the model. The model is now able to differentiate between the classes better and the precision and recall scores are higher.

We now have for the "BD" class 42% of the predicitons that are "BD" and 33% for "SALE"
