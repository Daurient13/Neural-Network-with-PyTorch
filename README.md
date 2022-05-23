# Neural-Network-with-PyTorch

# Dataset
in this project I use tabular data like the previous machine learning project. This data is talking about Diabetes, data consists of 768 rows and 9 columns.
Here are the column variables.

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

# Import Package
import common packages:

**import numpy as np**

**import pandas as pd**

from **sklearn.model_selection** import **train_test_split**

from **sklearn.pipeline** import **Pipeline**

from **sklearn.compose** import **ColumnTransformer**

from **jcopml.utils** import **load_model, save_model**

from **jcopml.pipeline** import **num_pipe, cat_pipe**

from **jcopml.plot** import **plot_missing_value**

from **jcopml.feature_importance** import **mean_score_decrease**

# Import Data

import data with pandas

# Explanation
unfortunately in PyTorch there is no preprocessor like in Scikit-Learn, because this is a tabular dataset, then in PyTorch still use Scikit-Learn, but if you have entered image data, text data, unstructured data it is very good with PyTorch.
