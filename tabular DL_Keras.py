import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from math import radians, cos, sin, asin, sqrt
import pydot
import seaborn as sns
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

label_column = 'price'
kc_raw_data = pd.read_csv('../data/kc_house_data.csv')

kc_raw_data['sale_yr'] = pd.to_numeric(kc_raw_data.date.str.slice(0,4))
kc_raw_data['sale_month'] = pd.to_numeric(kc_raw_data.date.str.slice(4,6))
kc_raw_data['sale_day'] = pd.to_numeric(kc_raw_data.date.str.slice(6,8))
kc_data = pd.DataFrame(kc_raw_data, columns=[
        'sale_yr','sale_month','sale_day', 'view', 'waterfront', 'lat', 'long',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
        'zipcode','sqft_living15','sqft_lot15','price'])

kc_data = kc_data.sample(frac=1)

train = kc_data.sample(frac=0.8)
'Train:' + str(train.shape)
validate = kc_data.sample(frac=0.1)
'Validate:' + str(validate.shape)
test = kc_data.sample(frac=0.1)
'Test:' + str(test.shape)

t_model = Sequential()
t_model.add(Dense(100, activation="relu", input_shape=(xsize,)))
t_model.add(Dense(50, activation="relu"))
t_model.add(Dense(ysize))
t_model.compile(
    loss="mean_squared_error",
    optimizer=Adam(lr=0.001),
    metrics=[metrics.mae])

epochs = 500
batch = 128

cols = list(train.columns)
cols.remove(label_column)
history = model.fit(
    train[cols], train[label_column],
    batch_size=batch,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    validation_data=(validate[cols],validate[label_column]),
    callbacks=keras_callbacks
)
score = model.evaluate(test[cols], test[label_column], verbose=0)

train_mean = train[cols].mean(axis=0)
train_std = train[cols].std(axis=0)
train[cols] = (train[cols] - train_mean) / train_std
validate[cols] = (validate[cols] - train_mean) / train_std
test[cols] = (test[cols] - train_mean) / train_std


def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r


FIXED_LONG = -122.213896
FIXED_LAT = 47.560053
kc_raw_data['distance'] = kc_raw_data.apply(lambda row: haversine(FIXED_LONG, FIXED_LAT, row['long'], row['lat']),
                                            axis=1)
kc_raw_data['greater_long'] = (kc_raw_data['long'] >= FIXED_LONG).astype(int)
kc_raw_data['less_long'] = (kc_raw_data['long'] < FIXED_LONG).astype(int)
kc_raw_data['greater_lat'] = (kc_raw_data['lat'] >= FIXED_LAT).astype(int)
kc_raw_data['less_lat'] = (kc_raw_data['lat'] < FIXED_LAT).astype(int)

data_zipcode = train_zipcode_x
data_main = train_otherfeatures_x
data_y = train_y

zipcode_input = Input(shape=(1,), dtype='int32', name='zipcode_input')
x = Embedding(output_dim=5, input_dim=200, input_length=1)(zipcode_input)
zipcode_out = Flatten()(x)
zipcode_output = Dense(1, activation='relu', name='zipcode_model_out')(zipcode_out)

main_input = Input(shape=(data_main.shape[1],), name='main_input')
lyr = keras.layers.concatenate([main_input, zipcode_out])
lyr = Dense(100, activation="relu")(lyr)
lyr = Dense(50, activation="relu")(lyr)
main_output = Dense(1, name='main_output')(lyr)

t_model = Model(
    inputs=[main_input, zipcode_input],
    outputs=[main_output, zipcode_output]
)
t_model.compile(
    loss="mean_squared_error",
    optimizer=Adam(lr=0.001),
    metrics=[metrics.mae],
    loss_weights=[1.0, 0.5]
)

from sklearn.manifold import TSNE
import seaborn as sns

zipcode_embeddings = model.layers[1].get_weights()[0]
labels = train_zipcode_x
zipcode_embeddings.shape
tsne_model = TSNE(perplexity=200, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(zipcode_embeddings)

x1 = []
y1 = []
avg_price1 = []
for index, value in enumerate(train_zipcode_x):
    zipcode = train_zipcode_x.iloc[index]
    price = train_y.iloc[index]
    avg_price1.append(price)
    x1.append(new_values[zipcode][0])
    y1.append(new_values[zipcode][1])

f, ax = plt.subplots(2, 1)

cmap = sns.cubehelix_palette(n_colors=10, start=0.3, rot=0.4, gamma=1.0, hue=1.0, light=0.9, dark=0.1, as_cmap=True)
axs0 = ax[0].scatter(x1, y1, s=20, c=avg_price1, cmap=cmap)
f.colorbar(axs0, ax=ax[0], orientation='vertical')