import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import *

#Initialization of parameters
x_train = np.array([[0.1, 0.2, 0.3 , 0.4], [0.3, 0.7, 0.6, 0.3]])
y_train = np.array([1.0, 0.5])
m = x_train.shape[0]
n = x_train.shape[1]
initial_w = np.zeros(n)
initial_b = 1.0
iterations = 10000
learningRate = 0.3
predicted = np.zeros(m)

#Somewhat standardizing input data
print(f'Initial parameters: {initial_w}, {initial_b}')

#Plot Data
# plt.scatter(x_train, y_train, marker='x', c='r') 
# plt.title("House area vs. House cost")
# plt.ylabel('Cost in 1000s of Euros')
# plt.xlabel('Area in m^2')
# plt.show()


#Compute Univariate Cost
cost = multivariateCost(x_train, y_train, initial_w, initial_b)
print(f'Cost at initial parameters: {cost:.3f}')

#Compute the Univariate gradient
tmp_dj_dw, tmp_dj_db = multivariateGradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial parameters:', tmp_dj_dw, tmp_dj_db)

#Perform Gradient Descent
w,b= multivariateGradientDescent(x_train ,y_train, initial_w, initial_b, 
                     multivariateCost, multivariateGradient, learningRate, iterations)
print("w,b found by gradient descent:", w, b)

#Calculate prediction data
# w = 194.595
# b = 108.815
for i in range(m):
    predicted[i] = np.dot(w,x_train[i]) + b
    print(f'Real Value: {y_train[i]}, Predicted: {predicted[i]}')

