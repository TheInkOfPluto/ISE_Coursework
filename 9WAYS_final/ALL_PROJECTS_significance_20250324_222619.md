## Statistical Significance Results (20250324_222619)

**ANOVA F-statistic**: 3.0517  
**p-value**: 0.0100


### Tukey HSD Post-hoc Test

| Group1 | Group2 | Mean Diff | p-adj | Lower | Upper | Reject |
|--------|--------|-----------|-------|-------|-------|--------|
| GaussianNB | KNeighborsClassifier | 0.0505 | 0.9931 | -0.1402 | 0.2412 | False |
| GaussianNB | LightGBM | 0.1506 | 0.2208 | -0.0402 | 0.3413 | False |
| GaussianNB | LinearSVC | 0.0389 | 0.9989 | -0.1518 | 0.2297 | False |
| GaussianNB | LogisticRegression | 0.0727 | 0.9374 | -0.118 | 0.2634 | False |
| GaussianNB | MultinomialNB | 0.1357 | 0.3432 | -0.055 | 0.3264 | False |
| GaussianNB | RandomForest | 0.0081 | 1.0 | -0.1827 | 0.1988 | False |
| GaussianNB | SGDClassifier | -0.0346 | 0.9995 | -0.2253 | 0.1561 | False |
| GaussianNB | XGBoost | 0.1659 | 0.1308 | -0.0248 | 0.3567 | False |
| KNeighborsClassifier | LightGBM | 0.1001 | 0.7248 | -0.0907 | 0.2908 | False |
| KNeighborsClassifier | LinearSVC | -0.0116 | 1.0 | -0.2023 | 0.1792 | False |
| KNeighborsClassifier | LogisticRegression | 0.0222 | 1.0 | -0.1685 | 0.2129 | False |
| KNeighborsClassifier | MultinomialNB | 0.0852 | 0.8604 | -0.1055 | 0.2759 | False |
| KNeighborsClassifier | RandomForest | -0.0424 | 0.9979 | -0.2332 | 0.1483 | False |
| KNeighborsClassifier | SGDClassifier | -0.0851 | 0.8611 | -0.2758 | 0.1056 | False |
| KNeighborsClassifier | XGBoost | 0.1154 | 0.556 | -0.0753 | 0.3062 | False |
| LightGBM | LinearSVC | -0.1116 | 0.5987 | -0.3024 | 0.0791 | False |
| LightGBM | LogisticRegression | -0.0779 | 0.9098 | -0.2686 | 0.1128 | False |
| LightGBM | MultinomialNB | -0.0149 | 1.0 | -0.2056 | 0.1759 | False |
| LightGBM | RandomForest | -0.1425 | 0.2829 | -0.3332 | 0.0482 | False |
| LightGBM | SGDClassifier | -0.1852 | 0.0627 | -0.3759 | 0.0055 | False |
| LightGBM | XGBoost | 0.0154 | 1.0 | -0.1754 | 0.2061 | False |
| LinearSVC | LogisticRegression | 0.0337 | 0.9996 | -0.157 | 0.2245 | False |
| LinearSVC | MultinomialNB | 0.0968 | 0.7583 | -0.094 | 0.2875 | False |
| LinearSVC | RandomForest | -0.0309 | 0.9998 | -0.2216 | 0.1599 | False |
| LinearSVC | SGDClassifier | -0.0735 | 0.9332 | -0.2643 | 0.1172 | False |
| LinearSVC | XGBoost | 0.127 | 0.4297 | -0.0637 | 0.3177 | False |
| LogisticRegression | MultinomialNB | 0.063 | 0.9721 | -0.1277 | 0.2537 | False |
| LogisticRegression | RandomForest | -0.0646 | 0.9677 | -0.2553 | 0.1261 | False |
| LogisticRegression | SGDClassifier | -0.1073 | 0.6472 | -0.298 | 0.0834 | False |
| LogisticRegression | XGBoost | 0.0933 | 0.7919 | -0.0975 | 0.284 | False |
| MultinomialNB | RandomForest | -0.1276 | 0.4232 | -0.3184 | 0.0631 | False |
| MultinomialNB | SGDClassifier | -0.1703 | 0.1115 | -0.361 | 0.0204 | False |
| MultinomialNB | XGBoost | 0.0302 | 0.9998 | -0.1605 | 0.221 | False |
| RandomForest | SGDClassifier | -0.0427 | 0.9978 | -0.2334 | 0.148 | False |
| RandomForest | XGBoost | 0.1579 | 0.1736 | -0.0329 | 0.3486 | False |
| SGDClassifier | XGBoost | 0.2005 | 0.033 | 0.0098 | 0.3913 | True |

### Paired t-tests vs Baseline (GaussianNB)

| Model | t-statistic | p-value | Significant (p<0.05) |
|--------|-------------|---------|----------------------|
| MultinomialNB | 3.6966 | 0.0209 | ✅ |
| LogisticRegression | 1.2721 | 0.2723 | ❌ |
| RandomForest | 0.2952 | 0.7826 | ❌ |
| LinearSVC | 0.7908 | 0.4733 | ❌ |
| SGDClassifier | -1.2154 | 0.2911 | ❌ |
| KNeighborsClassifier | 2.4060 | 0.0739 | ❌ |
| XGBoost | 4.0059 | 0.0161 | ✅ |
| LightGBM | 3.6505 | 0.0218 | ✅ |

*All tests based on [SciPy stats module](https://docs.scipy.org/doc/scipy/reference/stats.html)*
