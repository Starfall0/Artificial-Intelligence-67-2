import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

# Generate two clusters of data
X1, y1 = make_blobs(n_samples=100, n_features=2, centers=1,
                           center_box=(2.0, 2.0), cluster_std=0.75, random_state=69)

X2, y2 = make_blobs(n_samples=100, n_features=2, centers=1,
                           center_box=(3.0, 3.0), cluster_std=0.75, random_state=69)

# Combine data for normalization
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(100), np.ones(100)))

# Normalize z = (x - u) / s
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Creating a neural network model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=300, batch_size=200, verbose=0)

# Making predictions
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob).astype(int).ravel()

# Ploting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(xx.shape)

# Vitualization
fig = plt.figure()
# Fill the background with class regions
plt.contourf(xx, yy, Z, alpha=0.8, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

# Add the decision boundary as a black line
plt.contour(xx, yy, Z, levels=[0.5], colors=['black'], linewidths=1)

# Plot data points for Class 1 and Class 2
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', edgecolor='k', label="Class 1")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', edgecolor='k', label="Class 2")

# Labels, title, and legend
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Plane')
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.show()