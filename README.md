# Bachelor Thesis

## An evaluation of Interpretation-Nets applied to Logistic Regression for explaining Neural Networks

*Appendix A: Program Code*

### Setup

This code was tested on **python==3.9.12** using the dependencies of the **requirements.txt** file provided.

### Reproduce results of thesis

To reproduce the results of the thesis, run the following jupyter notebook scripts:

```
[instance]_1_generateData_n[numFeatures][addNoise]
[instance]_2_lambda_n[numFeatures][addNoise]
[instance]_3_inet_n[numFeatures][addNoise]
LR_4_eval_n[numFeatures][addNoise] (only for instance = 'LR')
```

Specify the experiment parameters using:
- *instance* = {'DT', 'LR'}, use either 'DT' (for inets for Decision Tree) or 'LR' (for inets for Logistic Regression, Plain Logistic Regression and Plain Decision Trees)
- *numFeatures* = {'5', '10', '20'}
- *addNoise* = {'-noise', ''}


For example:
```
LR_1_generateData_n10
LR_2_lambda_n10
LR_3_inet_n10
LR_4_eval_n10
```

```
DT_1_generateData_n5-noise
DT_2_lambda_n5-noise
DT_3_inet_n5-noise
```

Afterwards, view the results in "05-BA/data_LR" (if *instance*=LR) or "05-BA/data" (if *instance*=DT).