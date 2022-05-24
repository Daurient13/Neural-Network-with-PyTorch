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

![datasetloader](https://user-images.githubusercontent.com/86812576/169831023-1b6cbe36-26e0-4517-9142-fbab0ef5b2da.png)

First of all prepare a dataloader with minibatch per 64 data.

# Training Preparation -> MCO
- Model

I will make 3 hidden layers. The first hidden layer will consist of 16 neurons, the second hidden layer will consist of 8 neurons, and the third hidden layer will consist of 4 neurons. ReLu activation will be included in each hidden layer, and ended with a multiclass classification with logsoftmax.

- Criterion

In Criterion, NLLLoss will be used for multiclass classification, which models ending with logsoftmax.

- Optimizer

In the Optimizer, I use AdamW, namely Adaptive Momentum with Weight Decay, where there is a regulation to reduce overfit. With learning rate 0.001

# Training Loop
![Screenshot 2022-05-23 211503](https://user-images.githubusercontent.com/86812576/169839561-854192f3-6add-4001-8345-e3a9ed6bd7b2.png)

Prepare two lists, train_cost, and test_cost the goal is to see the loss goes down or not at once analyzed. Furthermore, in train_cost, it can be seen that the code above already has Feedforward, Calculate Loss, Backpropagation, update weight, and the total loss will be averaged to cost.

![Screenshot 2022-05-23 213848](https://user-images.githubusercontent.com/86812576/169844276-84d5497a-1268-4060-acc7-4b2c4513b56d.png)

What about test_cost?
The test data should never be fit (training) because it is a data leak meaning that in the test data we will never touch gradient, backprop, update weight and optimizer. but only use Feedforward because it is prediction. So the code is just feedforward and calculate the cost in order to get a history of loss reduction and overfit analysis. Basically the code is the same but a lot of things are thrown away.

the first in PyTorch there are tools that guarantee no gradient calculations, namely torch.no_grad(). If we set this it will save a lot of memory, because behind the scenes there is a lot of memory used for gradient calculations, while we will not calculate gradients also avoid data leaked. Status changed to model.eval() for evaluation, initialize cost with 0

By doing this we can predict the error in the test data

# Cost History

![image](https://user-images.githubusercontent.com/86812576/169942774-9868d6ce-48b9-4780-ac71-f6180a0985d7.png)

From the plot history above, it can be seen that 400 epochs are actually sufficient. If we plot the cost history then there is something that can be analyzed, here we can see that the training is getting smarter, but the test at some point actually goes up. This means that actually we only need up to 400 epochs because we always take the lowest test (cost). 

Can we go back to 400 epochs? of course not. then how do we find the best epoch? because currently the resulting model is at the very end of the line, and this model is classified as overfit.

# Predict
If we want to make predictions then just copy it in the testing code. The difference is that there is no need to use a dataloader, because it will always use the append function.

![Screenshot 2022-05-24 104322](https://user-images.githubusercontent.com/86812576/169944744-bebd314b-aa4f-4adc-a549-39503c1afd69.png)

the output above is not a prediction but logsoftmax, i.e. probability.
If you want predictive results, then take the highest probability. I just use the argmax function to get the index, so it generates predictions on the data test for a total of 154 predictions

# Accuracy
if you want to calculate accuracy, then compare the test data with the previous prediction which is exactly the same and then calculate the average. True means true, False means false. The accuracy result is 68%, which is not too bad.

# Reducing Overfit with Dropout
Is there anything else that can reduce the overfit besides regularization? Dropout is the easiest way to reduce overfit in neural networks invented by Geoffrey Hinton.
Here is the illustration.

![image](https://user-images.githubusercontent.com/86812576/169946724-24ff88c4-c0c8-4a87-a412-1d21a5c67a73.png)
the picture above is a photo of Mr Bean
We can see the picture on the right with lots of holes, maybe even half the picture is hollow, but we can still recognize that the picture is a photo of Mr Bean. If a neural network can recognize something even though there are parts that are omitted, it means that it can generalize. The neural network can recognize the image because it sees other features.

![image](https://user-images.githubusercontent.com/86812576/169947374-fd8f8f1c-48ea-4fa2-91d2-3821f7f597c9.png)

the idea is that in each iteration of each neuron there will be a probability that it will be eliminated. if we enter a dropout of 20%, that means there will be a percentage of 20% of neurons that will be dropped out. Tn the example above, after the first iteration, the dark colored neurons are affected by dropout. So in training, it's as if the infrastructure is 3-3-1, the temporary weight is multiplied by zero as well as the backprop.

So, it's like training a smaller neural network, but it's different in each iteration because each neuron has a 20% chance of dropping out. By using dropout, the machine will not only train one good feature, but when that feature is hit by a dropout, it will take advantage of other features to predict the same thing.

Another intuition about understanding dropouts is that if we dropout (remove features) it will add bias, if we increase bias then the variance decreases (Bias Variance Tradeoff).

# Next Model with Dropout

![Screenshot 2022-05-24 113400](https://user-images.githubusercontent.com/86812576/169949707-3ae06269-35e3-4eb0-bf5e-a42179d12205.png)

After linear, and activated, add a layer called Droput Layer by 20%. Don't dropout the output layer, because it will throw the prediction, the dropout is only on the hidden layer.

# Training
Training will be made more simple. The previous training code has a lot of repetition, therefore the following code is simplified like a Machine Learning Engineer.

![Screenshot 2022-05-24 113400](https://user-images.githubusercontent.com/86812576/169950620-dcd5bbf3-34f1-4c69-b8a5-931b1c03b571.png)

