# Explaining Neural Networks without Access to Training Data

Official implementation of the paper "Explaining Neural Networks without Access to Training Data" by Sascha Marton, Stefan Lüdtke Christian Bartelt, Andrej Tschalzev and Heiner Stuckenschmidt.

This repository contains the code to replicate the experiments from the paper. The used packages and corresponding versions can be found in the requirements.txt.
Replicating the requirements comprises three steps:
  1. Generate Datasets --> Execute 01_generate_data.ipynb for the corresponding number of variables and the predefined setting (We have created one file for each number of variables considered in the paper). To reproduce the results of the paper, you need to run all notebooks starting with "01"
  2. Train Networks Lambda --> Execute 02_lambda_net.ipynb for the corresponding number of variables and the predefined setting (We have created one file for each number of variables considered in the paper). To reproduce the results of the paper, you need to run all notebooks starting with "02"
  3. Train I-Net --> The Notebooks starting with "03" are used to train an I-Net and display all results. For the Case study, execute the files starting with "03_interpretation_net-n23" for each function family. To generate the results for the performance comparison on the real world datasets, please run the files starting "05_evaluate" for the corresponding function family. The visual comparison can be performed by running the notebooks starting with "03_interpretation_net-n02".

For the ablation study, the data generation is more elaborate. An expemplary file for the data generation is given in the corresponding notebooks "01b" and "02b". To reproduce the experiments, you need to run those notebooks for each number of variables by changing "number_of_variables" in the config at the top of the notebook to the corresponding value. This has to be performed for each function family (an expemplary notebook is given for each function family). Afterward, you can simply run the python scripty ending with "BENCHMARK" for the corresponding function family to reproduce the results of the ablation study.

The function families are identified in th code as follows:
    - vanilla --> standard decision tree
    - SDT1 --> univariate SDT
    - SDT-1 --> standard SDT

The network parameters used for training the I-Net have to be generated manually, since they are too large to upload. However, I can provide you with them upon request.
