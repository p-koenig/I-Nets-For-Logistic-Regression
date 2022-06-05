# InES_XAI

## Structure of the Project


## Setup

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Anaconda

This implementation is based on Python 3.6 and more, which is recommended to use to ensure a correct execution. However, other versions might work as well, but this was not investigated.

The easiest way to setup Python with a lot of commonly used packages is the installation of the Anaconda Distribution, which was also used for the development of this project.

#### Python Packages

Before a code execution additional python packages are required. For a fast installation the concept of requirements files is used. More information can be found here: https://pip.readthedocs.io/en/1.1/requirements.html 
The folder `99_helper_functions` contains two different requirements files:

* `official_project_requirements.txt` - created with the python package pipreqsnb (https://github.com/ivanlen/pipreqsnb). It only contains the specific packages required by the XAI project.

* `official_full_requirements.txt` - created with the pip freeze command and contains all packages installed in the development environment including packages that are not directly linked to the XAI project. This file is for backup only and for completeness reasons.

To install packages specified in a requirement text file, use the following command: 

    pip install -r path/to/requirements_file

Example: 

    pip install -r ./official_project_requirements.txt
    
Furthermore, two bash scripts are provided to ease the creation and installation of both previously introduced requirement files:

* `create_requirements.sh` - creates a `project_requirements.txt` and `full_requirements.txt` file.

* `install_requirements.sh` - installs the packages listed in the current `official_project_requirements.txt` or `official_full_requirements.txt` file depending on the user's input.

> Note : It might be necessary to activate admin rights to install new python packages.

# Explanations for Neural Networks by Neural Networks
Official implementation of the paper "Explanations for Neural Networks by Neural Networks" by Sascha Marton, Stefan Lüdtke and Christian Bartelt

To replicate the results from the paper, just replace the relevant parameters values in the config of each notebook by the values from the paper and leave all additional parameters unchanged. Please note, that the notebooks 01, 02 and 03 need to be run in subsequent order, while keeping the relevant parameters equal. The used libraries and versiona are contained in the requirements.txt.

The relevant parameters include:
Parameter     | Exemplary Value   | Explanation
------------- | ------------- | -------------
d | 3 | degree
n  | 5  | number of variables
sample_sparsity  | 5  | sparsity of the polynomial (has to be smaller than the maximum number of monomials for variable-degree combination)
polynomial_data_size  | 50,000  | number of functions to generate for lambda-net training
lambda_nets_total  | 50,000  | number of lambda-nets trained (lambda_nets_total <= polynomial_data_size)
lambda_dataset_size | 5,000  | number of samples per polynomial
lambda_dataset_size | 50,000  | number of trained lambda-nets used for the training of I-Net (interpretation_dataset_size <= lambda_nets_total))
noise | 0  | noise level
interpretation_net_output_monomials | 5 | max number of monomials contained in the polynomial predicted by the I-Net (usually equals the sample_sparsity)

