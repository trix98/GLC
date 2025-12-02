# Tutorials:

## Qucik Start

## Tutorial Example - Lipid class Prediction 

### GgmLipidClassifier (GLC)

The **GgmLipidClassifier (GLC)** is a workflow designed to systematically predict lipid classes directly from **MS1-only data** in untargeted LC–MS lipidomics.

GLC integrates two complementary sources of information within a unified scoring framework:

1. **Accurate-mass database matching:**  
   Features are tentatively assigned to candidate structures from the LIPID MAPS Structural Database (LMSD).

2. **Gaussian Graphical Models (GGM):**  
   A GGM is estimated from feature intensities, capturing conditional dependencies between features. Prior work [ref] and our own analyses [ref] suggest that these GGMs encodes lipid-class structure.

By combining the local network context provided by the GGM with tentative database matches, GLC generates refined predictions of lipid subclass.


This notebook showcases the use of GLC for lipid class prediction for untargeted lipidomic features for a reverse-phase lipidomics LC-MS assay. 
AddNeuroMed stuff:
X features, X samples

### Load Packages

```python
import pandas as pd
```

### Load Example Data

```python 
...
```

### Process Data
Don't forget to mention how can do a more detailed search - refer to other notebook

### Lipid Class Prediction 

### Evaluate 
Dosen't use annotations. Nonethless can test against
Add heatmaps