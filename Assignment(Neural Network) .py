#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

# Step function (activation function)
def step_function(x):
    return 1 if x >= 0.5 else 0

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        # Initialize weight and bias
        self.weight = np.random.rand()  # Single weight for 1D output
        self.bias = np.random.rand()    # Initialize bias randomly
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Predict output for a given input
    def predict(self, x):
        linear_output = x * self.weight + self.bias
        return step_function(linear_output)

    # Train the perceptron
    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                # Update weight and bias
                self.weight += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

# Training data (classification of numbers < 0.5 as 0, >= 0.5 as 1)
X = np.array([0.1, 0.3, 0.7, 0.9])
y = np.array([0, 0, 1, 1])

# Initialize and train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Function for user input classification
def user_input_classification():
    user_input = float(input("Enter a number between 0 and 1: "))
    if 0 <= user_input <= 1:
        prediction = perceptron.predict(user_input)
        print(f"Perceptron classification: {prediction}")
    else:
        print("Please enter a number between 0 and 1")

# Run the user input classification
user_input_classification()


# In[5]:


import numpy as np

# Step function (activation function)
def step_function(x):
    return 1 if x >= 0.5 else 0

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        # Initialize weight and bias
        self.weight = np.random.rand()  # Single weight for 1D output
        self.bias = np.random.rand()    # Initialize bias randomly
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Predict output for a given input
    def predict(self, x):
        linear_output = x * self.weight + self.bias
        return step_function(linear_output)

    # Train the perceptron
    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                # Update weight and bias
                self.weight += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

# Training data (classification of numbers < 0.5 as 0, >= 0.5 as 1)
X = np.array([0.1, 0.3, 0.7, 0.9])
y = np.array([0, 0, 1, 1])

# Initialize and train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Function for user input classification
def user_input_classification():
    user_input = float(input("Enter a number between 0 and 1: "))
    if 0 <= user_input <= 1:
        prediction = perceptron.predict(user_input)
        print(f"Perceptron classification: {prediction}")
    else:
        print("Please enter a number between 0 and 1")

# Run the user input classification
user_input_classification()


# In[6]:


def step_function(x):
    return 1 if x >= 0.5 else 0

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=200):
        # Initialize weight and bias
        self.weight = np.random.rand()  # Single weight for 1D output
        self.bias = np.random.rand()    # Initialize bias randomly
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Predict output for a given input
    def predict(self, x):
        linear_output = x * self.weight + self.bias
        return step_function(linear_output)

    # Train the perceptron
    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                # Update weight and bias
                self.weight += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

# Training data (classification of numbers < 0.5 as 0, >= 0.5 as 1)
X = np.array([0.1, 0.3, 0.7, 0.9])
y = np.array([0, 0, 1, 1])

# Initialize and train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Function for user input classification
def user_input_classification():
    user_input = float(input("Enter a number between 0 and 1: "))
    if 0 <= user_input <= 1:
        prediction = perceptron.predict(user_input)
        print(f"Perceptron classification: {prediction}")
    else:
        print("Please enter a number between 0 and 1")

# Run the user input classification
user_input_classification()


# In[7]:


def step_function(x):
    return 1 if x >= 0.5 else 0

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=200):
        # Initialize weight and bias
        self.weight = np.random.rand()  # Single weight for 1D output
        self.bias = np.random.rand()    # Initialize bias randomly
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Predict output for a given input
    def predict(self, x):
        linear_output = x * self.weight + self.bias
        return step_function(linear_output)

    # Train the perceptron
    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                # Update weight and bias
                self.weight += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

# Training data (classification of numbers < 0.5 as 0, >= 0.5 as 1)
X = np.array([0.1, 0.3, 0.7, 0.9])
y = np.array([ 0.49,0.51, 1, 1])

# Initialize and train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Function for user input classification
def user_input_classification():
    user_input = float(input("Enter a number between 0 and 1: "))
    if 0 <= user_input <= 1:
        prediction = perceptron.predict(user_input)
        print(f"Perceptron classification: {prediction}")
    else:
        print("Please enter a number between 0 and 1")

# Run the user input classification
user_input_classification()


# In[ ]:




