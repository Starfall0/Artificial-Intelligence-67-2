import numpy as np
import matplotlib.pyplot as plt
import keras.api.models as mod
import keras.api.layers as lay

# define parameters
# pitch: The periodicity of the sine-like wave
# step: The window size used to create sequences for the RNN
# N: Total number of data points
# n_train: Number of training data points (70% of total)
pitch = 20
step = 2 # change step to 2
N = 500
n_train = int(N*0.7)

# generate data function sawtooth wave
def gen_data(x):
    return (x % pitch)/pitch

# generate data
t = np.arange(1, N+1)
y = np.sin(0.05*t*10) + 0.8 * np.random.rand(N) # sine wave
y = np.array(y)

# plot original data
plt.figure()
plt.plot(y)
plt.show()

# convert data to matrix
def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])
    return np.array(X), np.array(Y)

train, test = y[0:n_train], y[n_train:N]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before): ", train.shape, test.shape)
print("Dimension (After):  ", x_train.shape, x_test.shape)

# create Recurrent neural network
model = mod.Sequential()
model.add(lay.SimpleRNN(units=64, input_shape=(step, 1), activation='relu')) # make units to 64 to make the model overfit
model.add(lay.Dense(units=1))  # output layer

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
hist = model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1) # change epoch to 100 to make the model overfit
prediction = model.predict(x_test)
plt.plot(hist.history['loss'])

# visualization
fig = plt.figure()
plt.plot(y_test, color='blue', label='Original Data')  # เส้น y

plt.plot(prediction , linestyle='--', color='red', label='Prediction')  # ทำเส้นประ prediction

plt.title('Comparison between Original Data and Prediction')
plt.xlabel('t')
plt.ylabel('x')
plt.grid(True, axis='both')
plt.legend(loc='upper right')  # add lines description
plt.show()
