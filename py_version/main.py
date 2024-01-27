import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import *

#Initialization of parameters

x_train = np.array([])
y_train = np.array([])
input = open("../housing-barcelona.csv", encoding="utf8")
input.readline()
istream = input.readline()
skip = False

while (len(istream) != 0):
    if(not(skip)):
        istream = istream.split(",")
        area = istream[2][0:len(istream[2])-3]
        cost = istream[0][0:len(istream[0])-1]
        print(area)
        print(cost)
        print(istream)
        cost = cost.split(".")
        cost = cost[0] + cost[1]
        x_train = np.append(x_train, [float(area)])
        y_train = np.append(y_train, [float(cost)])
    try:
        istream = input.readline()
    except Exception as e:
        print(e)
        skip = True



y_train = y_train / 1000
x_train = x_train / max(x_train)
m = x_train.shape[0]
initial_w = 2.0
initial_b = 1.0
iterations = 10000
learningRate = 0.3
predicted = np.zeros(m)

print(f'Initial parameters: {initial_w}, {initial_b}')

#Plot Data
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("x_train vs. y_train")
plt.ylabel('y_train')
plt.xlabel('x_train')
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
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()