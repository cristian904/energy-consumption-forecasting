import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, BatchNormalization, Activation
from keras.layers import LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

LOOK_BACK = 100
N_FEATURES = 8

dataset = pd.read_csv('train_electricity.csv')
y = dataset['Consumption_MW']
X = dataset.drop(['Consumption_MW', 'Date'], axis = 1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = list(y_test)
y_train = list(y_train)

print(X_train.shape)

features_set = []  
labels = []  
for i in range(LOOK_BACK, X_train.shape[0]):  
    features_set.append(np.reshape(X_train[i-LOOK_BACK:i], (LOOK_BACK*N_FEATURES,)))
    labels.append(y_train[i])
features_set, labels = np.array(features_set), np.array(labels)
print(features_set.shape)
features_set = np.reshape(features_set, (features_set.shape[0], LOOK_BACK, N_FEATURES)) 
print(features_set.shape)

model = Sequential()
model.add(Conv1D(filters=512, kernel_size=2, input_shape=(LOOK_BACK, N_FEATURES)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(features_set, labels, epochs = 50, batch_size = 64) 


test_features = []  
for i in range(LOOK_BACK, X_test.shape[0]):  
    test_features.append(np.reshape(X_test[i-LOOK_BACK:i], (LOOK_BACK, N_FEATURES,)))
test_features = np.array(test_features)  
test_features = np.reshape(test_features, (test_features.shape[0], LOOK_BACK, N_FEATURES)) 
predictions = model.predict(test_features)
print(predictions)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


plt.plot(y_test[LOOK_BACK:100+LOOK_BACK], color = 'black', label = 'Real data')
plt.plot(predictions[LOOK_BACK:100+LOOK_BACK], color = 'green', label = 'Predicted data')
plt.title('Energy consumption')
plt.xlabel('Time')
plt.ylabel('Energy consumption')
plt.legend()
plt.show()