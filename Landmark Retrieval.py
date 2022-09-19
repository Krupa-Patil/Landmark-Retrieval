# -*- coding: utf-8 -*-
"""ML_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q4nKjsQKxh-CG6fJ3dMznePfTGRTBxkH
"""

import tensorflow as tf 
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


from google.colab import drive
drive.mount('/content/drive')

train = "/content/drive/MyDrive/ADIV/ADIV_train"
test = "/content/drive/My Drive/ADIV/ADIV_test"
val = "/content/drive/My Drive/ADIV/ADIV_Validation"

SIZE_X = SIZE_Y = 224
batch_size=64
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    train,
    labels='inferred',
    label_mode = "categorical",
    color_mode='rgb',
    batch_size= batch_size,
    image_size=(SIZE_X,SIZE_Y),
    seed=123,
    validation_split=0.1,
    subset='training',
)
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    val,
    labels='inferred',
    label_mode = "categorical",
    color_mode='rgb',
    batch_size= batch_size,
    image_size=(SIZE_X,SIZE_Y),
    seed=123,
    validation_split=0.1,
    subset='validation',
)

def augment(x,y):
  image = tf.image.random_brightness(x, max_delta=0.05)
  return image,y

ds_train = ds_train.map(augment)

basemodel = VGG19(weights="imagenet", include_top=False,input_shape=(224,224,3))

basemodel.summary()

startmodel=basemodel.output
startmodel=Flatten(name='flatten')(startmodel)
startmodel=Dense(128,activation='relu')(startmodel)
startmodel=Dense(7,activation="softmax")(startmodel)
model = Model(inputs=basemodel.input, outputs=startmodel)

for layer in basemodel.layers:
    layer.trainable = False
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit_generator(ds_train,
validation_data = (ds_validation), 
epochs = 3
, verbose = 1,
callbacks=[es])

plt.plot(history.history['loss'],label = 'train_loss')
plt.plot(history.history['val_loss'], label = 'testing_loss')
plt.title('loss')
plt.legend()
plt.show()

result = model.evaluate(ds_validation,batch_size=128)
print("test_loss, test accuracy",result)

plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()



img_path = '/content/drive/MyDrive/ADIV/ADIV_test/MainBuilding_Circle/mainbui66 (3)_resized.jpg'

img = load_img(img_path, target_size=(224, 224))

x = img_to_array(img)
x = np.expand_dims(x, axis=0)

print('Input image shape:', x.shape)

my_image = imread(img_path)
imshow(my_image)

print(model.predict(x))

result=model.predict(x)
#ds_train.class_indices

results={
    0:'BT_Circle',
    1:'Cliet_Circle',
    2:'Gymkhana_Circle',
    3:'Hostel_Circle',
    4:'LHC_Circle',
    5:'MainBuilding_Circle',
    6:'Mech_Circle',
}

if result[0][0]==1:
  prediction = 'BT_Circle'
elif result[0][1]:
  prediction = 'Cliet_Circle'
elif result[0][3]:
   prediction = 'Gymkhana_Circle'
elif result[0][4]:
  prediction = 'Hostel_circle'
elif result[0][5]:
  prediction = 'LHC_Circle'
elif result[0][6]:
  prediction = 'MainBuilding_Circle'
elif result[0][7]:
  prediction = 'Mech_Circle'
else:
  prediction = 'wrong entry'

print(prediction)



img_path = '/content/drive/MyDrive/ADIV/ADIV_test/Gymkhana_Circle/.35_resized.jpg'

img = load_img(img_path, target_size=(224, 224))

x = img_to_array(img)
x = np.expand_dims(x, axis=0)

print('Input image shape:', x.shape)

my_image = imread(img_path)
imshow(my_image)

print(model.predict(x))

keras_file = "model1.h5"
keras.models.save_model(model,keras_file)



# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
