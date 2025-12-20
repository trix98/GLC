# GgmLipidClassifier (glc)

<div style="text-align: center;">
    <img src="./docs/images/glc_logo.png" alt="GLC_logo">
</div> 



GgmLipidClassifier (GLC) is an open-source Python package designed to systematically predict lipid class directly from MS1-only data in untargeted liquid chromatography–mass spectrometry (LC–MS) lipidomic workflows. GLC integrates accurate-mass database searching with Gaussian graphical models (GGM) estimated from feature intensities to predict lipid subclass and main class according to the LIPID MAPS Structural Database (LMSD) ontology. Requiring just a feature table, GLC predicts lipid class for most detected features and does not require prior annotation or MS2 for prediction. 

## Overview

### Sources and Materials 
Documentation is avaliable on our [Read the Docs page](https://trix98.github.io/GLC/).  
  
The package source code is accessible via GitHub at: https://github.com/trix98/GLC

# Installation 
You can install GLC from PyPI using pip:
```bash
pip install ggmlipidclassifier
```

# Issues and Colloboration
Thank you for supporting the GLC project. GLC is an open-source software and welcomes any form of contribution and support.

## Issues
Please submit any bugs or issues via the project's [GitHub page](https://github.com/trix98/GLC) issue page include any details about the (glc.__version__) together with any relevant input data/metadata. 

## Pull requests
You can actively colloborate on the GLC package by submitting any changes via a pull request. All pull requests will be reviewed by the GLC team and merged in due course. 

## Contributions
If you would like to become a contibuter on the GLC project, please contact Thomas Rix at tir21@ic.ac.uk 

# Acknowledgment 
This package was developed as part of Thomas Rix's PhD project at Imperial College London, supported by the European Union project
HUMAN (grant EC101073062, UKRI EP/X035840/1). It is free to use, published under BSD 3-Clause licence. 

The authors gratefully acknowledge Prof Simon Lovestone for permission to use the AddNeuroMed dataset.   
Dr María Gómez-Romero for help with dataset pre-processing.   
René Neuhaus and Sara Martínez for their support on the data treatment of the metformin-HIIE dataset.  

# Citing us
If you found this package useful, please consider citing us. 

## Publication 
The article is currently in the process of journal submission. 









