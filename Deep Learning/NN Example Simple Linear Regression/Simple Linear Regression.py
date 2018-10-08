import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

# Generate random data to train on
np.random.seed(0)

observations = 1000
xs = np.random.uniform(-10, 10, (observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

inputs = np.column_stack((xs, zs))

# Create targets we will aim at
noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2 * xs - 3 * zs + 5 + noise

# Initialize weights and bias
init_range = 0.1
weights = np.random.uniform(-init_range, init_range, (2, 1))
biases = np.random.uniform(-init_range, init_range, 1)

print(weights, biases)

# Set a learning rate
learning_rate = 0.02

# Train the model
for i in range(100):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas ** 2) / 2 / observations
    #print(loss)
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)    
    
# Print final weights and biases calculated
print(weights, biases)

    
    
    
    
    
    
    
    