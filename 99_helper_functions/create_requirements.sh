#!/bin/bash
pip install pipreqsnb
echo 'Create full list of installed python packages (independant of project)...'
pip freeze > ./full_requirements.txt 
echo 'full_requirements.txt created.'
echo 'Create list of installed python packages necessary for InES_XAI project...'
pipreqsnb ../ --savepath ./project_requirements.txt 
echo 'project_requirements.txt created.'
