# Neural-Network-with-PyTorch

# Dataset
in this project I use tabular data like the previous machine learning project. This data is talking about Diabetes, data consists of 768 rows and 9 columns.
Here are the column variables.

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI (Body Mass Index)
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
# Quick Exploratory Data Analysis
unfortunately in PyTorch there is no preprocessor like in Scikit-Learn, because this is a tabular dataset, then in PyTorch still use Scikit-Learn, but if you have entered image data, text data, unstructured data it is very good with PyTorch.

From the data above, there are several columns with missing values, 'Insulin' has a missing value of 48.70% which will be dropped from the column. While the other columns will be left as they are, including the 'SkinThickness' column which has a missing value of almost 30%

Then check whether the data is balanced. The percentage if we use the baseline accuracy is 65%, then this is not imbalanced data.

# Dataset Splitting
split the data into X, and y

X = all columns except the target column.

y = 'Outcome'.values as target, because PyTorch can't interact with pandas then change it to numpy

test_size = 0.2 (which means 80% for train, and 20% for test)

# Data Preprocessing
Just a reminder that in PyTorch there is no data preprocessor, so I still use Scikit-Learn.

Just like a machine learning project on a preprocessor, I will separate numeric data and categorical data for the purpose of impute data not for training. for that directly fit_transfrom on the processor and put it in the new X_train variable. when this step is done it will generate a numpy array so it must be converted to torch.tensor.

# Dataset & DataLoader
convert train data and test data to torch.tensor. But you have to be careful because the target column is a classification (0 and ) then you must use an integer torch.LongTensor, unless you use a sigmoid or binary classification ( 0 to 1) then you can use a FloatTensor. And in this case I will use logsoftmax so I will use integer. Next, so as not to forget to add to.device so that it can use the GPU if there is one.
