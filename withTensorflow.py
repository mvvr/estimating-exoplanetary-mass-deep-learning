# Importing required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv('exoplanet_data.csv')


# Removing NaN values
data = data.dropna()

# Splitting the dataset into inputs (X) and output (y)
X = data[["pl_bmassj", "pl_radj", "pl_orbper", "st_mass", "pl_eqt", "pl_pnum"]].astype(float)
y = y = data["pl_bmassj"].astype(float)

# Scaling the input values
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Creating the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, input_shape=(6,), activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Plotting the training and validation loss curves

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Testing the model
y_pred = model.predict(X_test)

# Plotting the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

# Evaluating the model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
