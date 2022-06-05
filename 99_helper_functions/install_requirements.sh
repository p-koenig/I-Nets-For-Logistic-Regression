#!/bin/bash
read -n1 -p "Do you want to install project specific packages only (recommended)? [y,n]" doit 
case $doit in  
  y|Y) pip install -r ./official_project_requirements.txt ;; 
  n|N) echo '' ; read -n1 -p "Do you want to install all packages (not recommended)? [y,n]" doit2 
    case $doit2 in  
        y|Y) pip install -r ./official_full_requirements.txt ;; 
        n|N) echo '' ; echo Nothing to install. Done. ;; 
        *) echo '' ; echo Wrong input. Please try again. ;; 
    esac ;; 
  *) echo '' ; echo Wrong input. Please try again. ;; 
esac
