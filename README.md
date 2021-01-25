# **Project: Mercedes-Benz Greener Manufacturing**

### In this project a Machine Learning algorithm is developed to predict the time a car will spend on the test bench based on vehicle configuration. The intention is that an accurate model will be able to reduce the total time spent testing vehicles by allowing cars with similar testing configurations to be run successfully. This is an example of a machine learning regression and supervised learning task since, it requires predicting a continuous target variable (the duration of test) based on the model of the car [X1-X9].

### The dataset and other details can be downloaded from kaggle website: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing

## Overview
**The goals / steps of this project are the following:**
The over all problem statement is solved with two different approach. First, by considering it as a regression problem, second, by converting the dataset into a classification problem

1. Understanding Dataset
2. Data Visualization
3. Feature Selection
4. Implementing Regression Analysis with 10-fold cross validation
5. Performing Grid search to find best parameters for above models
6. Implementing a Classification model
7. Predicting duration of test performed on vehicles using test dataset

[//]: # (Image References)

[image1]: ./Graphs/Graph_1.png
[image2]: ./Graphs/
[image3]: ./Graphs/
[image4]: ./Graphs/Features.png

## Brief explanation of each step

### Understanding Dataset

- This dataset also brings the challenge to tackle the curse of dimensionality.
- The data set encompasses around 373 attributes and 8400 instances.
- There are 2 files available to use. ‘train.csv’ and ‘test.csv’ both containing 4209 rows of data. These rows essentially indicate the number of vehicles provided to us and their various configurations.
- The ‘train.csv’ also contains the target variable which is available as the first column in ‘Y’ which are continuous numeric values in seconds.
- Both the test and train contain 8 different vehicle features with names such as X0 – X8.
> All the features have been anonymized and they do not come with any physical representation.
> The description of data does not indicate that vehicle features are configuration options.
- There are 8 categorical features with values encoded as strings such as ‘A’,’B’, ‘C’, etc.
- There are 368 tests which are integer values and only indicate 0 or 1 i.e it indicates whether the tests are performed or not.

### Data Visualization

We plotted various different attributes for the mean time they contribute to and some of them had relatively high contribution to the total time and some of them had very low contribution. This led us to the analysis of PCA and ICA where attributes who contributed less would eventually be dropped or not considered for training.

![alt_text][image1]
![alt_text][image2]
![alt_text][image3]

The below graph shows us the important features selected by XGBoost. The features selected are categorical features as well as binary features. In our case, we will give more importance to the categorical features and hence, we go ahead and plot the mean times for all the categorical features. The categorical features are X5, X0, X8, X6, X1, X3, X2.
![alt_text][image4]

### Data Preprocessing

Since we have 8 categorical columns and 369 integer values. These 369 integer values are already in '0' and '1' format, so we will convert other 8 categorical columns into ‘One Hot Encoded’ columns to have entire train data in same format. The ‘One Hot Encoding’ is also applied to the ‘Test’ dataset. After having data in common format, we apply various types of feature selection techniques on the dataset.

### Feature Selection

Three feature selection techniques are utilised: PCA, SVD, ICA
1. **Principal component analysis (PCA)** is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. It emphasizes variation and brings out strong patterns in a dataset. PCA is implemented as a transformer object that learns n-components in its ‘fit’ method, and can be used on new data to project it on these components.

2. **Singular Value Decomposition (SVD)** transformer performs linear dimensionality reduction, contrary to PCA. This estimator does not center the data before computing the singular value decomposition. SVD makes it easy to eliminate the less important parts of the representation to produce the approx representation with any desired number of dimensions.

3. **Independent component analysis (ICA)** is a statistical and computational technique for revealing hidden factors that underlie sets of random variables, measurements, or signals. ICA defines a generative model for the observed multivariate data, which is typically given as a large database of samples. In the model, the data variables are assumed to be linear or nonlinear mixtures of some unknown latent variables, and the mixing system is also unknown. ICA can be seen as an extension to principal component analysis and factor analysis. ICA is a much more powerful technique, however, capable of finding the underlying factors or sources when these classic methods fail completely.

### Model Implementation: Regression
The primary evaluation metric for this dataset is the co-efficient of determination (R^2 measure). R^2 is the measure of quality of a model that is used to predict one continuous variable from a number of other variables. It describes the amount of variation in the dependent variable, in this case the testing time of vehicles in seconds, based on independent variables which, in this case is the combination of vehicle custom feature.

Regression models utilized to test the prediction of test times were:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regression
5. KNN Regressor
6. Artificial Neural Network (ANN)- Keras

For Artificial Neural Network (ANN)- Keras

- The ANN methodology using Keras is a way to organize layers.
- There are many activation functions to choose from and we started with **‘Relu’** since, the time in seconds or the ‘Y’ columns doesn’t have any negative values. However, with some trial and error, the best option was to select the **‘Sigmoid’** function.
- The **‘Kernal Initializers’** define the way, to set the initial random weights of keras layers. We have used **‘Kernal initializers = normal’**.
- We have used **4 layers with 500, 300, 150 and 75 neurons** each. The last layer is the output layer with 1 neuron which is to  estimate the ‘time in seconds’ and consists of 1 column only.
- An objective function is required to compile the model and we have used **‘mean_squared_error’** as an indication of score. The optimizer used is **‘Adam’** which is an algorithm for first-order gradient based optimization of stochastic objective function.
- The variable ‘train_x’ is used to store the training dataset on which the ‘train_y’ values are fit. This will be further used to predict values. The **validation split is 0.05** which is 5% of the 4207 rows available.
- The number of epochs used were, found by trial and error. It was primarily set to be 100, but beyond 30, it is trying to overfit the train set. Hence we stopped it at 30.

### Model Implementation: Classification
1. Logistic Regression
As we know that the ground truth is available as a continuous variable. And so we decided to create different bins and consider them as different classes.


### Results
The train scores after implementing different regression models:

**Using PCA:**

|     **Model**             | **R^2 value** | **MSE** |
|---------------------------|---------------|---------|
| Linear Regression         | 0.5357        | 76.2435 |
| Ridge Regression          | 0.5483        | 74.2217 |
| Lasso Regression          | 0.5460        | 74.6026 |
| Decision Tree Regression  | 0.1300        | 181.2991|
| KNN Regressor             | 0.051         | 167.8394|


**Using TSVD:**

|     **Model**             | **R^2 value** | **MSE** |
|---------------------------|---------------|---------|
| Linear Regression         | 0.4927        | 83.06134|
| Ridge Regression          | 0.4928        | 86.06116|
| Lasso Regression          | 0.4927        | 83.06133|
| Decision Tree Regression  | 0.0344        | 155.2708|
| KNN Regressor             | 0.0517        | 167.2846|

|     **Model**             | **Accuracy**  | **MSE** |
|---------------------------|---------------|---------|
| Logistic Regression       | 0.3352        | -       |
| Artificial Neural Network | 0.5639        | 59.1438 |

## Different Approach

- Implementation of ANN is really flexible, and hence we could have fine tuned, activation functions with different number of neurons and layers. This was very time consuming for 500 epochs and hence wasn’t tried comprehensively.
- We could have implemented one or more variable reduction techniques to the ANN and logistic model. The variable reduction techniques weren’t tried on ANN and Logistic regression as ANN is supposed to be learning at a better rate. We would like to think that with relevant attributes it could have worked well.
- The classification of logistic model could have been done in a more comprehensive way to increase sensitivity or ‘classification’ of data. Since the data is highly skewed to the right, the classification is biased.
- Also increasing the sensitivity of the less classified data was challenging. This could have been accomplished by oversampling and under sampling techniques.
- We could have used ‘XGBoost’ model for improving the optimizing process.
- Other models like Random Forest, Bagging, Boosting, and Gradient Boosting can be tested for accuracy.

