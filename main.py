# -*- coding: gb2312 -*-
import numpy as np
from causallearn.search.Granger.Granger import Granger
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, concatenate, Layer, Flatten, Reshape, Attention
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from scipy.stats import skew


# Define a custom Graph Convolutional Layer
class GraphConv(Layer):
    """
    Custom Graph Convolutional Layer for spatial data.
    """

    def __init__(self, units, A, activation=None, kernel_size=2, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.A = A  # Adjacency matrix
        self.activation = tf.keras.activations.get(activation)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Initialize weights
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(GraphConv, self).build(input_shape)

    def call(self, inputs):
        # Perform graph convolution operation
        output = tf.matmul(self.A, inputs)  # Multiply input by adjacency matrix
        output = tf.matmul(output, self.kernel) + self.bias  # Apply linear transformation
        if self.activation is not None:
            output = self.activation(output)  # Apply activation function
        return output


# Construct a geospatial adjacency matrix
def construct_adjacency_matrix(I, J):
    """
    Constructs an adjacency matrix for a grid graph of size I x J.
    """
    N = I * J  # Total number of nodes in the grid
    A = np.eye(N)  # Start with an identity matrix
    for i in range(I):
        for j in range(J):
            index = i * J + j
            # Connect left and right nodes
            if j > 0:
                A[index][index - 1] = 1  # left
            if j < J - 1:
                A[index][index + 1] = 1  # right
            # Connect top and bottom nodes
            if i > 0:
                A[index][(i - 1) * J + j] = 1  # top
            if i < I - 1:
                A[index][(i + 1) * J + j] = 1  # bottom
    return A


# Construct a causal adjacency matrix
def granger_test(risk_matrix):
    """
    Constructs the Granger causality matrix based on input time series data.
    """
    N = len(risk_matrix)
    granger = np.ones((N, N))  # Initialize with ones
    for i in range(N):
        k1 = risk_matrix[i]
        for j in range(N):
            k2 = risk_matrix[j]
            # If any series has only zeros, set causality to 1
            if np.all(k1 == 0) or np.all(k2 == 0):
                granger[i][j] = 1
            else:
                G = Granger()
                data = np.array([k1, k2]).T  # Prepare data for Granger test
                p_value_matrix = G.granger_test_2d(data)
                coeff = G.granger_lasso(data)  # Apply Lasso to get coefficients
                granger[i][j] = max(coeff[1])  # Assign max coefficient
                print(i, j)
    return granger


# Load and preprocess data
Value = np.load('traffic_risk.npy')  # Load the test data
A = construct_adjacency_matrix(10, 10)  # Construct adjacency matrix
A_2 = granger_test(Value)  # Compute the Granger causality matrix
X = Value.T  # Transpose the data for further processing
A_3 = A * A_2  # Combine adjacency matrix and Granger causality matrix

# Replace NaN and Inf values with valid numbers
X[np.isnan(X)] = -1
X[np.isinf(X)] = -1
X[X < 0] = np.exp(X[X < 0])

# Feature extraction using sliding window (window size = 3, step size = 1)
activation_function = 'relu'
window_size = 3
step_size = 1
features = []

# Extract features for each window
for i in range(0, len(X) - window_size + 1, step_size):
    window = X[i:i + window_size]
    # Compute various features
    mean = np.mean(window, axis=0)
    variance = np.var(window, axis=0)
    peak = np.max(np.abs(window), axis=0)
    amplitude = np.ptp(window, axis=0)
    skewness = skew(window, axis=0)
    # Concatenate all features into a single vector
    concatenated_features = np.concatenate([mean, variance, peak, amplitude, skewness])
    features.append(concatenated_features)

features = np.array(features)  # Convert list of features to numpy array

# Split data into training, validation, and test sets (60:20:20 ratio)
train_size = int(0.6 * len(features))
val_size = int(0.2 * len(features))
test_size = len(features) - train_size - val_size

train_x, val_x, test_x = features[:train_size], features[train_size:train_size + val_size], features[
                                                                                            train_size + val_size:]
train_y, val_y, test_y = X[window_size - 1:train_size + window_size - 1], X[
                                                                          train_size + window_size - 1:train_size + val_size + window_size - 1], X[
                                                                                                                                                 train_size + val_size + window_size - 1:]


# Define the model architectures
def spatial_model(x):
    """
    Defines the spatial model using Graph Convolutional Layers.
    """
    model = Sequential()
    model.add(GraphConv(128, A=A, activation=activation_function, kernel_size=2))
    model.add(GraphConv(128, A=A, activation=activation_function, kernel_size=2))
    model.add(GraphConv(100, A=A, activation=activation_function, rnel_size=2))
    return model


def time_model(x):
    """
    Defines the temporal model using GRU layers.
    """
    model = Sequential()
    model.add(GRU(units=128, activation=activation_function, return_sequences=True))
    model.add(GRU(units=128, activation=activation_function, return_sequences=True))
    model.add(GRU(units=100, activation=activation_function))
    return model


def time_and_spatial_model(x):
    """
    Defines the combined spatio-temporal model using Graph Convolutional Layers.
    """
    model = Sequential()
    model.add(GraphConv(128, A=A_3, activation=activation_function, kernel_size=2))
    model.add(GraphConv(128, A=A_3, activation=activation_function, kernel_size=2))
    model.add(GraphConv(100, A=A_3, activation=activation_function, kernel_size=2))
    return model


def merge_model(x):
    """
    Merges the spatial, temporal, and spatio-temporal models into a single model
    and applies an attention mechanism for feature fusion.
    """
    x = Input(shape=(x.shape[1],))

    # Feature extraction using three different models
    model_1 = time_model(x)
    model_2 = spatial_model(x)
    model_3 = time_and_spatial_model(x)

    # Concatenate outputs of all models
    merged_features = concatenate([model_1.output, model_2.output, model_3.output])

    # Feature Fusion using Attention Mechanism
    attention_output = Attention()([merged_features, merged_features])  # Self-attention for feature fusion

    # Fully connected layers for prediction
    output = Dense(64, activation='tanh')(attention_output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=x, outputs=output)

    # Model Interpretability Analysis using Gradient Sensitivity Analysis
    grad_input = Flatten()(merged_features)  # Flatten the integrated feature matrix
    grad_output = Reshape((X.shape[1], X.shape[2]))(grad_input)  # Reconstruct the matrix
    grad_output = Dense(X.shape[1], activation='linear')(grad_output)  # Predict future risk values

    model_interpretability = Model(inputs=x, outputs=grad_output)

    return model, model_interpretability


# Custom RMSE loss function
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Compile and Train the Model
x = Input(shape=(train_x.shape[1],))
model, model_interpretability = merge_model(x)
model.compile(optimizer=Adam(learning_rate=0.001), loss=rmse, metrics=['mse', 'accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(val_x, val_y))

# Evaluate the model
test_loss, test_mse, test_acc = model.evaluate(test_x, test_y)
print(f"Test RMSE: {test_loss}, Test MSE: {test_mse}, Test Accuracy: {test_acc}")

# # Perform Model Interpretability Analysis
# grads = tf.gradients(model.output, model.input)[0]
# grads_fn = tf.keras.backend.function([model.input], [grads])
# grad_values = grads_fn([test_x])[0]

# # Construct the Grids Association Matrix G using the gradients
# G = np.dot(grad_values.T, A_3)
# print("Grids Association Matrix G: ", G)
