import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import *

# Function created to read in specific type of data
def readDataBarcelona():
    input = open("../housing-barcelona.csv", encoding="utf8")
    input.readline()
    istream = input.readline()
    skip = False
    x_train = np.array([])
    y_train = np.array([])
    while (len(istream) != 0):
        if(not(skip)):
            istream = istream.split(",")
            area = istream[2][0:len(istream[2])-3]
            cost = istream[0][0:len(istream[0])-1]
            cost = cost.split(".")
            cost = cost[0] + cost[1]
            x_train = np.append(x_train, [float(area)])
            y_train = np.append(y_train, [float(cost)])
        try:
            istream = input.readline()
        except Exception as e:
            print(e)
            skip = True
    return x_train, y_train


#Initialization of parameters
x_train, y_train = readDataBarcelona()
m = x_train.shape[0]
initial_w = 2.0
initial_b = 1.0
iterations = 10000
learningRate = 0.3
predicted = np.zeros(m)

#Somewhat standardizing input data
y_train = y_train / 1000
x_train = x_train / max(x_train)
print(f'Initial parameters: {initial_w}, {initial_b}')

#Plot Data
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("House area vs. House cost")
plt.ylabel('Cost in 1000s of Euros')
plt.xlabel('Area in m^2')
plt.show()


#Compute Univariate Cost
cost = computeUnivariateCost(x_train, y_train, initial_w, initial_b)
print(f'Cost at initial parameters: {cost:.3f}')

#Compute the Univariate gradient
tmp_dj_dw, tmp_dj_db = computeUnivariateGradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial parameters:', tmp_dj_dw, tmp_dj_db)

#Perform Gradient Descent
w,b,_,_ = univariateGradientDescent(x_train ,y_train, initial_w, initial_b, 
                     computeUnivariateCost, computeUnivariateGradient, learningRate, iterations)
print("w,b found by gradient descent:", w, b)

#Calculate prediction data
# w = 194.595
# b = 108.815
for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot the data and linear fit
plt.plot(x_train, predicted, c = "b")
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("House Area vs. House Cost")
plt.ylabel('Cost in 1000s of Euros')
plt.xlabel('Area in m^2')
plt.show()