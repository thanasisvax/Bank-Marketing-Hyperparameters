# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
The dataset consists of data about a marketing campaign over the phone. There are various information of the clients such as age, education, marital status etc. and if they have decided to subscribe into the marketing offer/package or not.
The goal of this problem was to predict if a client will subscribe or not into the marketing package. It is a classification problem in order to predict if a client will subscribe into the package based on the information available for the client (age, education, marital status etc.).

The best performing model was identified by the Automl and it was a Voting Ensemble model with accuracy close to 0.9174. However, the logistic regression model using hyperdrive achieved an accuracy of 0.90 which is pretty close to it.

## Scikit-learn Pipeline

The data are in a csv format which are represented by a table of data with their associated headers. It was used the TabularDatasetFactory function in order to create our dataset in the azure ml studio. After that there is a clean function which cleans the data (for instance the column marital transforms it to 0 or 1 based on if you are married or not). The data are processed from the clean function in order to be ready to be fed into the hyperdrive run. The logistic regression model is used in the train.py in order to predict 'y' column in the dataset. 

I have used the RandomParameterSampling in order to random select values for the tags "C" and "max_iter" which are the two parameters which need to be tuned in order to identify the best values which lead into the best performing model. For the max_iter tag, I used a choice of several values and for the C tag, I used a uniform distribution from values from 0.1-1. The benefits for this sampler is that there will be a random select of values for the tags in order to identify the best hyperparameters selection.

I have also used the BanditPolicy(slack_factor = 0.1, evaluation_interval=2) as a termination policy which every two iterations checks if the primary metric which is the accuracy falls outside the top 10% range. This policy helps to save time from continuing to explore hyperparameters that do not show promise of matching our primary metric.

The primary metric is the accuracy and I am trying to maximize it using the train.py model which is the logistic regression. I achieved an accuracy of 0.90 with the following best hyperparameters: ['--C', '0.3232054328858635', '--max_iter', '80']


## AutoML

From the AutoML, I received the Voting Ensemble model as the best run and fitted model for this dataset and classification problem. The weights amd various hyperparameters are shown for the best model. The accuracy metric was 0.9174. We can see that the l1-ratio is 0.38 and the max_iters is 1000 which are similar to the logistic regression hyperparameters above.

## Pipeline comparison
There is no significant difference in the accuracy between the two models as it can be seen by comparing the accuracies received from above. I believe that the logistic regression achieves a high accuracy and it is a simple model and it does not require a huge compute power.

## Future work

There is an ALERT on the automl model run that the data are biased which are leaned towards the no answer on the 'y' column. So, I believe a future improvement will be to remove this bias using various techniques.

