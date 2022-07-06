# Bachelor Thesis

## An evaluation of Interpretation-Nets applied to Logistic Regression for explaining Neural Networks

*Appendix A: Program Code*

### Setup

This code was tested on **python==3.9.12** using the dependencies of the **requirements.txt** file provided.

### Reproduce results of thesis

To reproduce the results of the thesis, run the following jupyter notebook scripts:

instance = {DT, LR} 
use DT for inets for Decision Tree
use LR for inets for Logistic Regression, Plain Logistic Regression and Plain Decision Trees

numFeatures = {5, 10, 20}

addNoise = {-noise, }

- \[instance\]_1_generateData_n\[numFeatures\]\[addNoise\]
- \[instance\]_2_lambda_n\[numFeatures\]\[addNoise\]
- \[instance\]_3_inet_n\[numFeatures\]\[addNoise\]

For example:

LR_1_generateData_n10

LR_2_lambda_n10

LR_3_inet_n10