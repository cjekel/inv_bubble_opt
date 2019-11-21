import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 512*2
epochs = 1000

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 3)

# load the XY data
with h5py.File('10k28x28data.hdf5', 'r') as f:
    a_group_key = list(f.keys())[0]
    X = f[a_group_key][()]
scaler = MinMaxScaler()
Y = np.load('mydoe.npy')
# Y = scaler.fit_transform(Y)
# Y = Y[:, :2]
num_predictions = Y.shape[1]

print('Original shape:', X.shape, Y.shape)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,
                                                    random_state=42)

print('Train shape:', X_train.shape, y_train.shape)

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.025))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))

model.add(Dropout(0.025))
model.add(Dense(num_predictions, activation='linear'))

model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['mse'])

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test))
# compute mae in each component
y_hat = model.predict(X_test)
e = np.abs(y_hat-y_test)
score = e.mean(axis=0)
print('E1 mae:', score[0])
print('E2 mae:', score[1])
if num_predictions == 3:
    print('G12 mae:', score[2])

# compute mean absolute percentage error
e_m = np.abs((y_hat - y_test)/y_test)
e_mape = e_m.mean(axis=0)
print('E1 mape:', score[0]*100)
print('E2 mape:', score[1]*100)
if num_predictions == 3:
    print('G12 mape:', score[2]*100)

emax = np.max(e, axis=0)
print('E1 max absolute error:', emax[0])
print('E2 max absolute error:', emax[1])
if num_predictions == 3:
    print('G12 max absolute error:', score[2])


plt.figure()
plt.plot(history.epoch, history.history['loss'], label='Training')
plt.plot(history.epoch, history.history['val_loss'], label='Validation')
plt.xlabel('epochs')
plt.ylabel('Mean absolute error')
plt.legend()
plt.grid()
plt.show()
# model.fit(x_train, y_train,
#           epochs=epochs,
#           batch_size=128)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])