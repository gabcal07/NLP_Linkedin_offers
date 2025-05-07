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

#### SMOTE additon

          precision    recall  f1-score   support

        ACCT      0.447     0.661     0.533       977
         ADM      0.463     0.502     0.482       954
        ADVR      0.189     0.500     0.274       120
        ANLS      0.195     0.187     0.191       791
         ART      0.210     0.310     0.250       310
          BD      0.420     0.554     0.478      2722
        CNSL      0.187     0.298     0.230       446
        CUST      0.333     0.306     0.319       841
        DIST      0.153     0.238     0.186        84
        DSGN      0.177     0.171     0.174       450
         EDU      0.322     0.521     0.398       426
         ENG      0.394     0.549     0.459      2393
         FIN      0.355     0.438     0.392      1594
        GENB      0.393     0.319     0.352       385
        HCPR      0.806     0.903     0.851      3293
          HR      0.578     0.756     0.655       550
          IT      0.550     0.266     0.359      5052
         LGL      0.636     0.798     0.708       481
        MGMT      0.340     0.135     0.193      4096
        MNFC      0.385     0.495     0.433      3532
        MRKT      0.343     0.419     0.377      1072
        OTHR      0.473     0.383     0.423      2459
          PR      0.127     0.200     0.155       260

...
accuracy 0.402 41156
macro avg 0.313 0.405 0.336 41156
weighted avg 0.421 0.402 0.386 41156

---

### Notes on GridSearchCV:

The optimal model is not that good but an issue is that there are a lot of classes and a lot of confusion between the classes, e.g for "BD" class 43% percent of predicitons are "SALE".
This means that the Naive Bayes model struggles to differentiate between the classes.

#### Adding the "Title" column:

We have a noticeably better score with the "title" column added to the model. The model is now able to differentiate between the classes better and the precision and recall scores are higher.

We now have for the "BD" class 42% of the predicitons that are "BD" and 33% for "SALE"

#### Using SMOTE:

Using SMOTE (Sampling Method for Imbalanced Learning) the model has a worst score than the model without it.

---

### Logistic Regression:

Same TF-IDF vectorizer but now we use a Logistic Regression model instead of a Naive Bayes model.

#### Test with Logistic Regression and SVD + SMOTE:

            precision    recall  f1-score   support

        ACCT      0.440     0.633     0.519       977
         ADM      0.503     0.525     0.514       954
        ADVR      0.165     0.517     0.251       120
        ANLS      0.126     0.149     0.136       791
         ART      0.090     0.184     0.121       310
          BD      0.411     0.574     0.479      2722
        CNSL      0.162     0.386     0.228       446
        CUST      0.317     0.384     0.347       841
        DIST      0.130     0.417     0.198        84
        DSGN      0.100     0.136     0.115       450
         EDU      0.170     0.239     0.199       426
         ENG      0.399     0.562     0.466      2393
         FIN      0.367     0.375     0.371      1594
        GENB      0.255     0.444     0.324       385
        HCPR      0.877     0.900     0.888      3293
          HR      0.592     0.791     0.677       550
          IT      0.642     0.235     0.344      5052
         LGL      0.609     0.827     0.702       481
        MGMT      0.276     0.126     0.173      4096
        MNFC      0.375     0.401     0.388      3532
        MRKT      0.366     0.390     0.377      1072
        OTHR      0.598     0.510     0.551      2459
          PR      0.074     0.127     0.094       260

...
accuracy 0.390 41156
macro avg 0.296 0.395 0.320 41156
weighted avg 0.427 0.390 0.380 41156

### Why NB is better than Logistic Regression ?

When you switch from a MultinomialNB to a LogisticRegression on high‑dimensional, highly‑sparse text data (TF–IDF) with 35 very imbalanced classes, it’s actually pretty common to see NB outperform—even though LR is a more “powerful” discriminative model—because:

**Data sparsity & feature noise**

MultinomialNB makes a simple “bag‑of‑words” assumption and smooths aggressively, so it’s extremely robust when each class has relatively few examples and most features never occur.

LogisticRegression tries to learn a weight for every feature‑class pair. With 15 000+ TF–IDF features and only a handful of samples in many of your 35 classes, it simply doesn’t have enough signal to tune all those weights—and it overfits or underfits noisily.

**Dimensionality reduction with TruncatedSVD**

You’re squashing ~15 000 dimensions down to 200. That can help with dense numeric data, but on sparse TF–IDF it often throws away the very positional/term‑specific clues that NB relies on.

In practice, TruncatedSVD on text is mostly used to speed up really large pipelines or to visualize. In a one‑step LogisticRegression you’re usually better off feeding the raw TF–IDF (or at least a higher SVD dimension like 1 000+).

**SMOTE on SVD projections**

SMOTE creates synthetic data by interpolating between real points in your 200‑dimensional SVD space. But there’s no guarantee those “in‑between” points correspond to a real TF–IDF distribution, so you’re injecting a lot of garbage samples.

With high‑dimensional text it’s often better to either rely on class_weight='balanced' (which you already do) or to do simple oversampling (e.g. random‑upsampling of the minority classes before vectorizing).

### Why Saga solver for Logistic Regression ?

Handles sparse matrices efficiently

Supports true multinomial classification

Can use regularization (l2, elasticnet, etc.)

Is scalable and parallelizable

### TF-IDF + SGDClassifier (no SVD or SMOTE):

           precision    recall  f1-score   support

        ACCT      0.480     0.700     0.569       977
         ADM      0.467     0.515     0.490       954
        ADVR      0.162     0.500     0.244       120
        ANLS      0.201     0.209     0.205       791
         ART      0.199     0.371     0.259       310
          BD      0.417     0.493     0.452      2722
        CNSL      0.183     0.321     0.233       446
        CUST      0.299     0.288     0.293       841
        DIST      0.082     0.417     0.138        84
        DSGN      0.211     0.184     0.197       450
         EDU      0.300     0.577     0.395       426
         ENG      0.414     0.546     0.471      2393
         FIN      0.394     0.444     0.417      1594
        GENB      0.332     0.332     0.332       385
        HCPR      0.781     0.921     0.845      3293
          HR      0.561     0.805     0.662       550
          IT      0.602     0.308     0.408      5052
         LGL      0.588     0.844     0.693       481
        MGMT      0.365     0.130     0.192      4096
        MNFC      0.414     0.463     0.437      3532
        MRKT      0.389     0.414     0.401      1072
        OTHR      0.528     0.380     0.442      2459
          PR      0.170     0.292     0.215       260

...
accuracy 0.413 41156
macro avg 0.323 0.436 0.351 41156
weighted avg 0.440 0.413 0.402 41156

---

### Feed Forward Neural Network:

#### FFNN (TF-IDF + SVD + SMOTE + StandardScaler):

Iteration 39, loss = 1.13582682
Validation score: 0.640863
Validation score did not improve more than tol=0.000100 for 6 consecutive epochs. Stopping.

precision recall f1-score support

        ACCT      0.452     0.588     0.511       977
         ADM      0.368     0.481     0.417       954
        ADVR      0.116     0.375     0.177       120
        ANLS      0.151     0.198     0.171       791
         ART      0.105     0.261     0.150       310
          BD      0.410     0.658     0.505      2722
        CNSL      0.147     0.296     0.196       446
        CUST      0.258     0.333     0.290       841
        DIST      0.118     0.321     0.173        84
        DSGN      0.098     0.158     0.121       450
         EDU      0.273     0.383     0.319       426
         ENG      0.388     0.489     0.433      2393
         FIN      0.367     0.346     0.356      1594
        GENB      0.216     0.377     0.275       385
        HCPR      0.830     0.876     0.853      3293
          HR      0.575     0.695     0.629       550
          IT      0.542     0.203     0.295      5052
         LGL      0.630     0.775     0.695       481
        MGMT      0.405     0.425     0.415      4096
        MNFC      0.401     0.176     0.245      3532
        MRKT      0.324     0.381     0.350      1072
        OTHR      0.465     0.391     0.425      2459
          PR      0.113     0.165     0.135       260
        PRCH      0.113     0.211     0.147       109
        PRDM      0.267     0.570     0.364       277
        PRJM      0.286     0.524     0.370       764
        PROD      0.077     0.213     0.114       150
          QA      0.319     0.629     0.423       334
        RSCH      0.223     0.401     0.286       599
        SALE      0.377     0.028     0.053      4312
         SCI      0.105     0.253     0.148       146
        STRA      0.064     0.204     0.097       235
        SUPL      0.077     0.137     0.099       226
        TRNG      0.243     0.262     0.253       461
         WRT      0.121     0.231     0.159       255

    accuracy                          0.378     41156

macro avg 0.286 0.372 0.304 41156
weighted avg 0.417 0.378 0.362 41156

---

### Side Notes

Regression with SVD and SMOTE not really good, what are possible reasons ?

- Data is too sparse, too many dimensions
- Simply NB (and maybe LogisticRegression) not suited for this task
- Too many classes
- Bad data preprocessing ?
- Not enough data, need more columns
- Logistic Regression:
  - Bad solver
  - Bad preprocessing (SVD + SMOTE)
  - Bad parameters ?

---
