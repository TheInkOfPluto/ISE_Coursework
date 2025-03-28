## Statistical Significance Results (20250327_173147)

**ANOVA F-statistic**: 2.9314  
**p-value**: 0.0125


### Tukey HSD Post-hoc Test

| Group1 | Group2 | Mean Diff | p-adj | Lower | Upper | Reject |
|--------|--------|-----------|-------|-------|-------|--------|
| GaussianNB | KNeighborsClassifier | 0.0505 | 0.9931 | -0.1403 | 0.2413 | False |
| GaussianNB | LightGBM | 0.1506 | 0.2214 | -0.0403 | 0.3414 | False |
| GaussianNB | LinearSVC | 0.0389 | 0.9989 | -0.1519 | 0.2298 | False |
| GaussianNB | LogisticRegression | 0.0727 | 0.9376 | -0.1182 | 0.2635 | False |
| GaussianNB | MultinomialNB | 0.1357 | 0.344 | -0.0551 | 0.3265 | False |
| GaussianNB | RandomForest | 0.0008 | 1.0 | -0.19 | 0.1917 | False |
| GaussianNB | SGDClassifier | -0.0215 | 1.0 | -0.2124 | 0.1693 | False |
| GaussianNB | XGBoost | 0.1659 | 0.1313 | -0.0249 | 0.3568 | False |
| KNeighborsClassifier | LightGBM | 0.1001 | 0.7254 | -0.0908 | 0.2909 | False |
| KNeighborsClassifier | LinearSVC | -0.0116 | 1.0 | -0.2024 | 0.1793 | False |
| KNeighborsClassifier | LogisticRegression | 0.0222 | 1.0 | -0.1687 | 0.213 | False |
| KNeighborsClassifier | MultinomialNB | 0.0852 | 0.8608 | -0.1056 | 0.276 | False |
| KNeighborsClassifier | RandomForest | -0.0497 | 0.9938 | -0.2405 | 0.1412 | False |
| KNeighborsClassifier | SGDClassifier | -0.072 | 0.9404 | -0.2629 | 0.1188 | False |
| KNeighborsClassifier | XGBoost | 0.1154 | 0.5569 | -0.0754 | 0.3063 | False |
| LightGBM | LinearSVC | -0.1116 | 0.5995 | -0.3025 | 0.0792 | False |
| LightGBM | LogisticRegression | -0.0779 | 0.9101 | -0.2687 | 0.1129 | False |
| LightGBM | MultinomialNB | -0.0149 | 1.0 | -0.2057 | 0.176 | False |
| LightGBM | RandomForest | -0.1498 | 0.2272 | -0.3406 | 0.0411 | False |
| LightGBM | SGDClassifier | -0.1721 | 0.1047 | -0.363 | 0.0187 | False |
| LightGBM | XGBoost | 0.0154 | 1.0 | -0.1755 | 0.2062 | False |
| LinearSVC | LogisticRegression | 0.0337 | 0.9996 | -0.1571 | 0.2246 | False |
| LinearSVC | MultinomialNB | 0.0968 | 0.7589 | -0.0941 | 0.2876 | False |
| LinearSVC | RandomForest | -0.0381 | 0.999 | -0.229 | 0.1527 | False |
| LinearSVC | SGDClassifier | -0.0605 | 0.9783 | -0.2513 | 0.1304 | False |
| LinearSVC | XGBoost | 0.127 | 0.4306 | -0.0638 | 0.3178 | False |
| LogisticRegression | MultinomialNB | 0.063 | 0.9722 | -0.1278 | 0.2539 | False |
| LogisticRegression | RandomForest | -0.0719 | 0.9413 | -0.2627 | 0.119 | False |
| LogisticRegression | SGDClassifier | -0.0942 | 0.7835 | -0.2851 | 0.0966 | False |
| LogisticRegression | XGBoost | 0.0933 | 0.7925 | -0.0976 | 0.2841 | False |
| MultinomialNB | RandomForest | -0.1349 | 0.3517 | -0.3257 | 0.056 | False |
| MultinomialNB | SGDClassifier | -0.1572 | 0.1778 | -0.3481 | 0.0336 | False |
| MultinomialNB | XGBoost | 0.0302 | 0.9998 | -0.1606 | 0.2211 | False |
| RandomForest | SGDClassifier | -0.0224 | 1.0 | -0.2132 | 0.1685 | False |
| RandomForest | XGBoost | 0.1651 | 0.1352 | -0.0257 | 0.356 | False |
| SGDClassifier | XGBoost | 0.1875 | 0.0574 | -0.0034 | 0.3783 | False |

### Paired t-tests vs Baseline (GaussianNB)

| Model | t-statistic | p-value | Significant (p<0.05) |
|--------|-------------|---------|----------------------|
| MultinomialNB | 3.6966 | 0.0209 | ✅ |
| LogisticRegression | 1.2721 | 0.2723 | ❌ |
| RandomForest | 0.0309 | 0.9769 | ❌ |
| LinearSVC | 0.7908 | 0.4733 | ❌ |
| SGDClassifier | -0.8686 | 0.4341 | ❌ |
| KNeighborsClassifier | 2.4060 | 0.0739 | ❌ |
| XGBoost | 4.0059 | 0.0161 | ✅ |
| LightGBM | 3.6505 | 0.0218 | ✅ |

*All tests based on [SciPy stats module](https://docs.scipy.org/doc/scipy/reference/stats.html)*
