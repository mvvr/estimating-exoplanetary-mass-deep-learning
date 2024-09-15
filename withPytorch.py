# Importing required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

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

# Converting data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train.values).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test.values).float()

# Defining the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initializing the neural network
net = Net()

# Defining the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Training the neural network
train_losses = []
test_losses = []

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = net(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    with torch.no_grad():
        y_pred_test = net(X_test)
        test_loss = criterion(y_pred_test, y_test)
        test_losses.append(test_loss.item())

    if epoch % 100 == 0:
        print('Epoch {}, Training Loss {}, Test Loss {}'.format(epoch, loss.item(), test_loss.item()))

# Testing the neural network
y_pred = net(X_test)

# Converting PyTorch tensors to numpy arrays
y_test = y_test.numpy()
y_pred = y_pred.detach().numpy()

# Estimating planetary mass
X_new = [[1.6, 1.2, 365, 1.5, 280, 2]]
X_new = sc_X.transform(X_new)
X_new = torch.from_numpy(X_new).float()
y_new = net(X_new)
y_new = y_new.detach().numpy()
print('Estimated Planetary Mass:', y_new[0][0])

# Evaluating the neural network performance


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# Plotting the training and testing losses
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

# Plotting the actual vs. predicted values
# Plotting the actual vs. predicted values with a line of best fit
plt.scatter(y_test, y_pred)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test.ravel(), y_pred.ravel(), 1))(np.unique(y_test)), color='red') #To fix this error, you can reshape the input arrays to be one-dimensional using the .ravel() method or the flatten() method. For example, you can modify the line of code that is causing the error to:
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Planetary Mass')
plt.show()

# Creating a histogram of the actual and predicted values
plt.hist(y_test, bins=30, alpha=0.5, label='Actual')
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted')
plt.legend()
plt.xlabel('Planetary Mass')
plt.ylabel('Frequency')
plt.title('Histogram of Actual and Predicted Planetary Mass')
plt.show()

# Creating a scatter plot of the actual vs. predicted values for each input feature
features = ['Planet Mass', 'Planet Radius', 'Orbital Period', 'Stellar Mass', 'Equilibrium Temperature', 'Number of Known Planets']
for i in range(6):
    plt.scatter(X_test[:, i], y_test, label='Actual')
    plt.scatter(X_test[:, i], y_pred, label='Predicted')
    plt.legend()
    plt.xlabel(features[i])
    plt.ylabel('Planetary Mass')
    plt.title('Actual vs. Predicted Planetary Mass for ' + features[i])
    plt.show()





# Creating a confusion matrix
y_test_class = (y_test > 0.5).astype(int)
y_pred_class = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test_class, y_pred_class)
print('Confusion Matrix:')
print(cm)

# Creating a classification report
target_names = ['Low Mass', 'High Mass']
print('Classification Report:')
print(classification_report(y_test_class, y_pred_class, target_names=target_names))


# Creating a ROC curve
fpr, tpr, thresholds = roc_curve(y_test_class, y_pred)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculating the area under the curve (AUC)
auc = roc_auc_score(y_test_class, y_pred)
print('AUC:', auc)

