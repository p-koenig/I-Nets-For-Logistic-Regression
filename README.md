# InES_XAI

## Structure of the Project

#### Important Folders
* `_baselib` - Contains fundamental python modules that implement basic functions imported by higher-level jupyter notebooks.
* `_setup` - Contains information regarding the required python packages to successfully execute code.
* `lahoffma` - Subrepository: contains resources managed exclusively by Lars Hoffmann. 
* `smarton` - Subrepository: contains resources managed exclusively by Sascha Marton.


## Setup

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Anaconda

This implementation is based on Python 3.6 and more, which is recommended to use to ensure a correct execution. However, other versions might work as well, but this was not investigated.

The easiest way to setup Python with a lot of commonly used packages is the installation of the Anaconda Distribution, which was also used for the development of this project.

#### Python Packages

Before a code execution additional python packages are required. For a fast installation the concept of requirements files is used. More information can be found here: https://pip.readthedocs.io/en/1.1/requirements.html 
The folder `_setup` contains two different requirements files:

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