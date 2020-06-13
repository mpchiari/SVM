# SVM
Implementing SVM using different gradient methods (mini-batch, stochastic, etc.) to predict if a person has or doesn't have diabetes.

To estimate the w, b of the soft margin SVM, we can minimize the loss function:
<p align="center">
  <img src="https://github.com/mpchiari/SVM/blob/master/images/svmFormula1.png" width="650" title="SVM formula">
</p>

In order to minimize the function, we first obtain the gradient with respect to wj, the jth item in the vector w, as follows:
<p align="center">
  <img src="https://github.com/mpchiari/SVM/blob/master/images/svmGradient.png" width="450" title="Gradient formula">
</p>

where:
<p align="center">
  <img src="https://github.com/mpchiari/SVM/blob/master/images/svmGradient2.png" width="450" title="Gradient formula">
</p>

As well as the gradient with respect to b:
<p align="center">
  <img src="https://github.com/mpchiari/SVM/blob/master/images/svmGradient3.png" width="350" title="Gradient formula">
</p>

where:
<p align="center">
  <img src="https://github.com/mpchiari/SVM/blob/master/images/svmGradient4.png" width="350" title="Gradient formula">
</p>

Command line to run the program: ```python3 SVM_gradient_descent.py data.txt```

## Data

Data.txt contains the data description (first 13 lines of the file), and a matrix with 9 columns and 760 rows. Columns 1-8 are input features, and column 9 is the class label. The features present in the dataset are the following: number of times pregnant, plasma glucose concentration a 2 hours in an oral glucose tolerance test, diastolic blood pressure (mm Hg), triceps skin fold thickness (mm), 2-Hour serum insulin (mu U/ml), body mass index (weight in kg/(height in m)^2), diabetes pedigree function, age (years), class (-1 tested negative, +1 tested positive).

## Algorithms
### Batch Gradient Descent
### Stochastic Gradient Descent
### Mini Batch Gradient Descent

## Tasks

Implement the different algorithms, run them on the given dataset and plot the value of the cost function Jk(w,b) vs. the number of iterations (k). Report the total time taken for convergence by each of the gradient descent techniques.

## Results
<p align="center">
  <img src="https://github.com/mpchiari/SVM/blob/master/images/results.png" width="750" title="Gradient formula">
</p>

Time batch gradient descent (in sec): 0.00840 <br/> 
Time stochastic gradient descent (in sec): 0.2467 <br/>
Time mini gradient descent (in sec): 0.07650 <br/>

<br/>
<br/>
<br/>
Class project for Machine Learning - Spring 2020
