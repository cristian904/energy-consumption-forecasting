import pandas as pd
from keras.models import model_from_json
import numpy as np

LOOK_BACK = 100
N_FEATURES = 8

dataset = pd.read_csv('train_electricity.csv')
y = dataset['Consumption_MW']
X = dataset.drop(['Consumption_MW', 'Date'], axis = 1)
last_X = X[-LOOK_BACK:]
validation_dataset = pd.read_csv('test_electricity.csv')
validation_dataset1 = validation_dataset.drop(['Date'], axis=1)
val = last_X.append(validation_dataset1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
val = sc.fit_transform(val)
val_features = []  
for i in range(LOOK_BACK, val.shape[0]):  
    val_features.append(np.reshape(val[i-LOOK_BACK:i], (LOOK_BACK, N_FEATURES,)))
val_features = np.array(val_features)  
val_features = np.reshape(val_features, (val_features.shape[0], LOOK_BACK, N_FEATURES)) 
json_file = open('model.json', 'r')
model_json = json_file.read()
model = model_from_json(model_json)
json_file.close()
model.load_weights('model.h5')
predictions = model.predict(val_features)
print(predictions.shape)

df = pd.DataFrame()
df['Date'] = validation_dataset['Date']
df['Consumption_MW'] = list(np.reshape(predictions, (1, predictions.shape[0]))[0])
df.to_csv("predictions.csv", index=False)
