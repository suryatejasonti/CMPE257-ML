# The MIT License (MIT)

# Copyright (c) 2014 Jake Cowton

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# A Neural Network to calculate if an RGB value is more red or blue
from perceptron import Perceptron

BLUE = 1
RED = 0
# Lowest MSE
LMSE = 0.001

def normalise(data):
    """
    MUST BE CUSTOMISED PER PROJECT
    Turn data into values between 0 and 1
    @param data list of lists of input data and output e.g.
        [
            [[0,0,255], 1],
            ...
        ]
    @returns Normalised training data
    """
    max = 0.0
    for i in data:
        for j in i[1:-1]:
            max = j if j > max else max
    
    temp_list = []
    for entry in data:
        entry_list = []
        for value in entry[1:-1]:
            # Normalise the data. 1/255 ~ 0.003921568
            entry_list.append(float(value*max))
        temp_list.append([entry_list, entry[-1]])
    return temp_list

def main(data):

    # Normalise the data
    training_data = normalise(data)

    # Create the perceptron
    p = Perceptron(len(training_data[0][0]))

    # Number of full iterations
    epochs = 0

    # Instantiate mse for the loop
    mse =999

    while (abs(mse-LMSE) > 0.002):
    #while epochs < 10000:
        # Epoch cumulative error
        error = 0

        # For each set in the training_data
        for value in training_data:

            # Calculate the result
            output = p.result(value[0])

            # Calculate the error
            iter_error = value[1] - output

            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.weight_adjustment(value[0], iter_error)

        # Calculate the MSE - epoch error / number of sets
        mse = float(error/len(training_data))

        # Print the MSE for each epoch
        print "The MSE of %d epochs is %.10f" % (epochs, mse)

        # Every 100 epochs show the weight values
        if epochs % 100 == 0:
            print "0: %.10f - 1: %.10f - 2: %.10f error: %.03f" % (p.w[0], p.w[1], p.w[2], mse**2)

        # Increment the epoch number
        epochs += 1
    return p
 
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False 

#generate a data set of 20. 
#for simplicity, 10 in the first quadrant, another 10 in the third quadrant 
X1 = []
Y1 = []
X2 = []
Y2 = []

for i in range(10):
    X1.append(random.uniform(0,1))
    Y1.append(random.uniform(0,1))
    X2.append(random.uniform(-1,0))
    Y2.append(random.uniform(-1,0))
x = np.array(X1)+X2
y=Y1+Y2
#label the data
data1 = [np.array([1,X1[i],Y1[i],1]) for i in range(10)]
data2 = [np.array([1,X2[i],Y2[i],-1]) for i in range(10)]
data = data1 + data2

p = main(data)
#print p.new_recall()



