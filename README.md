# IBM_Machine_Learning_Professional
This repo contains all the projects done during the IBM Machine Learning Professional Certificate on Coursera. I can really recommend this course to everyone who wants to create more portfolio projects and wants to learn a lot 
about some typical industry machine learning problems. Each course first explains some basics, dives deeper into some real problems and gives solutions to them plus hints and tricks. At the end of each course, you have to 
proove your skills by applying the learning techniques on a real world dataset of your choice. You have to solve a problem related to the course and have to create a final project report. The real world projects took the most 
time in this course, but I learned a lot through them. 

## Exploratory Data Analysis
The goal of the first assignment is to systematically explore any data. I decided to dive deeper into the Stroke Prediction Dataset from Kaggle (https://www.kaggle.com/fedesoriano/stroke-prediction-dataset). 
The notebook within the Exploratory_Data_Analysis folder explores the data set, executes some feature engineering and finally stores a pre-processed training and testing data set. These can later be used for directly performing some machine learning on them.
The documentation report of this project can be found in the documentation folder.

## Regression
In the second assignment, a regression problem was to be solved. I used the chocolate bar ratings dataset from Kaggle (https://www.kaggle.com/rtatman/chocolate-bar-ratings). The notebook "Exploratory_Data_Analysis.ipynb" performs some data exploration, feature
engineering and pre-processing of the dataset. The notebook "Train_and_Predict.ipynb" uses the prepared data to train and compare different regression techniques. 

Trained regression algorithms:
1) Linear Regression (used as baseline)
2) Ridge Regression
3) Random Forest Regressor
4) Gradient Boosting Regressor

The documentation report of this project can be found in the documentation folder.

## Classification
The goal of the third assignment was to solve a classification problem with machine learning. Here, I used the water quality dataset from Kaggle (https://www.kaggle.com/adityakadiwal/water-potability). The notebook "EDA_and_PreProcessing.ipynb" contains the exploratory data 
analysis and the pre-processing of the dataset. The notebook "Run_Classification.ipynb" contains the code for training different classification models and comparing their performance. 

Trained classification algorithms:
1) Logistic Regression (used as baseline)
2) Decision Tree Classifier
3) Support Vector Machine Classifier
4) Nearest Neighbor Classifier
5) Random Forest Classifier
6) Ada Boost Classifier

The documentation report of this project can be found in the documentation folder.

## Clustering
The goal of the fourth assignment was to solve a machine learning problem with unsupervised learning. I decided to apply K-means clustering on the customer clustering dataset from Kaggle (https://www.kaggle.com/dev0914sharma/customer-clustering). The notebook "Customer_Clustering.ipynb"
contains the full code (including exploratory data analysis, data pre-processing and the clustering training). 

The documentation report of this project can be found in the documentation folder.

## Deep Learning
In the assignment of the Deep Learning course, the goal was to develop a deep neural network to solve an image classification. As dataset, I used the Standford cars dataset (https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The notebook "Check_Augmentations.ipynb" 
loads an image and evaluates different augmentation techniques with the goal to find suited augmentations for the image classification model. The notebook "DeepLearning_Project.ipynb" contains the code for creating the input pipeline, 
comparing different sate-of-the-art convolutional neural networks, comparing different image sizes, comparing over-sampling to imabalnced learning and searching hyperparameters with Bayesian search.

Compared convolutional neural networks:
1) DenseNet121
2) Xception
3) ResNet50
4) EfficientNetB0
5) ResNet34
6) ResNet18

The documentation report of this project can be found in the documentation folder.

## Time Series
The last assignments goal was to analyze time series data and to perform some forecasting on this time series data. For this assignment, I used the superstore sales dataset from Kaggle (https://www.kaggle.com/rohitsahoo/sales-forecasting). The notebook "Exploratoy_Data_Analysis.ipynb"
contains the initial exploratory data analysis and some nice figures. The notebook "Time_Series_Analysis.ipynb" contains code for analyzing the daily sales data, verifying that the series is stationary and training and comparing different time series data models. 
The goal was to only use the daily sales data of the past to make a one week forecast. The root mean squared error is then used to compare the different time series techniques.

Time series models trained:
1) ARIMA
2) Simple RNN
3) LSTM

The documentation report of this project can be found in the documentation folder.
