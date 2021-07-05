### Run InceptionResNet V2 on 2000 full images

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import gc
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
from functools import partial
import sklearn.metrics as metrics

import tensorflow as tf
import keras
from keras.optimizers import Adam, Nadam, SGD, Adagrad
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint

import kerastuner as kt

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

tf.compat.v1.set_random_seed(seed_value)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

### Input parameters
batchSize = 20

output_dir = '~\\ModelsAllImage_2\\'

earlyStoppingCrit = {'monitor': 'val_accuracy',
                     'patience': 5,
                     'min_delta': 0.05,
                     'mode': 'max',
                     'restore_best_weights': True}
epochsCrit = 50

### Import Images
importPath = '~\\Image Data\\AllImageSplit'
imageSize = (270, 570) #(135, 285) #(1350, 2850)
batchSize = 20


train_ds = image_dataset_from_directory(
  importPath,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=imageSize,
  batch_size=batchSize)

test_ds = image_dataset_from_directory(
  importPath,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=imageSize,
  batch_size=batchSize)

class_names = ['Excellent', 'Shadow']

### Split out validation dataset
train_batches = tf.data.experimental.cardinality(train_ds)
val_ds = train_ds.take(train_batches // 4)
train_ds = train_ds.skip(train_batches // 4)
print('Number of Training batches: %d' % tf.data.experimental.cardinality(train_ds))
print('Number of Validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of Test batches: %d' % tf.data.experimental.cardinality(test_ds))

### Data Optimisation
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


def TuneNetwork(hp, inputNet):

    ### Initialise the parameters to search
    learning_rateIn = hp.Float('learning_rateIn', 1e-4, 1e-2, step=1e-6)
    activationIn = hp.Choice('activationIn', ['relu', 'sigmoid'])
    optimiserIn = hp.Choice('optimiserIn', ['Adam', 'SGD', 'Adagrad', 'Nadam'])
    DropOutIn = hp.Float('DropOutIn', 0.1, 0.9, step=0.05)

    ### Clear Backend session
    keras.backend.clear_session()

    ### Data Preprocessing
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

    ### Define Input Shape
    imageShape = imageSize + (3,)

    ### Define base model
    base_model = inputNet(input_shape=imageShape,
                          include_top=False,
                          weights='imagenet')

    ### Fix Pretrained Model weights
    base_model.trainable = False

    ### Pool pre-trained model outputs together
    global_average_layer = GlobalAveragePooling2D()

    ### Generate a fully connected Layer
    Dense_layer = Dense(1024, activation=activationIn)

    ### Generate predictions
    prediction_layer = Dense(1, activation='sigmoid')

    ### Link model together
    inputs = tf.keras.Input(shape=imageShape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(DropOutIn)(x) ### use a dropout layer to regularise
    #x = Dense_layer(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)

    # compile the model (should be done *after* setting layers to non-trainable)
    opt = eval(optimiserIn + '(lr=learning_rateIn)')

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    return model


def TestNetwork(inputNet, learning_rateIn, activationIn, optimiserIn, DropOutIn, tuner_epochs, tuner_initial_epoch, tuner_bracket,
                tuner_round):

    ### Clear Backend session
    keras.backend.clear_session()

    ### Data Preprocessing
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

    ### Define Input Shape
    imageShape = imageSize + (3,)

    ### Define base model
    base_model = inputNet(input_shape=imageShape,
                          include_top=False,
                          weights='imagenet')

    ### Fix Pretrained Model weights
    base_model.trainable = False

    ### Pool pre-trained model outputs together
    global_average_layer = GlobalAveragePooling2D()

    ### Generate a fully connected Layer
    Dense_layer = Dense(1024, activation=activationIn)

    ### Generate predictions
    prediction_layer = Dense(1, activation='sigmoid')

    ### Link model together
    inputs = tf.keras.Input(shape=imageShape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(DropOutIn)(x)  ### use a dropout layer to regularise
    #x = Dense_layer(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)

    # compile the model (should be done *after* setting layers to non-trainable)
    opt = eval(optimiserIn + '(lr=learning_rateIn)')

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    return model

ResRD = kt.RandomSearch(
            partial(TuneNetwork,inputNet = tf.keras.applications.InceptionResNetV2),
            objective='val_accuracy',
            seed=123,
            max_trials=10,
            executions_per_trial=1,
            overwrite=True)

early_stopping2 = EarlyStopping(**earlyStoppingCrit)

model_checkpoint_callback = ModelCheckpoint(filepath=output_dir+'InceptionResNetV2_Random.hdf5',
                                            save_weights_only=True,
                                            monitor='val_accuracy',
                                            mode='max',
                                            save_best_only=True)
ResRD.search(train_ds,
             validation_data=val_ds,
             epochs=epochsCrit,
             callbacks=[early_stopping2, model_checkpoint_callback])

Best_Random_HP = ResRD.get_best_hyperparameters()[0].values
print(Best_Random_HP)

model_checkpoint_callback = ModelCheckpoint(filepath=output_dir+'InceptionResNetV2_RandomFinal_epoch{epoch:02d}.hdf5',
                                            save_weights_only=True,
                                            monitor='val_accuracy',
                                            mode='max',
                                            save_best_only=False)
RandomHistory = TestNetwork(inputNet = tf.keras.applications.InceptionResNetV2,
                            **Best_Random_HP,
                            tuner_epochs = 0,
                            tuner_initial_epoch = 0,
                            tuner_bracket = 0,
                            tuner_round = 0).fit(train_ds,
                                                 validation_data=val_ds,
                                                 epochs=epochsCrit,
                                                 callbacks=[model_checkpoint_callback])

### Plot performance
acc = RandomHistory.history['accuracy']
val_acc = RandomHistory.history['val_accuracy']

loss = RandomHistory.history['loss']
val_loss = RandomHistory.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.xticks(np.arange(len(RandomHistory.history['accuracy'])), np.arange(1, len(RandomHistory.history['accuracy'])+1))
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.xticks(np.arange(len(RandomHistory.history['accuracy'])), np.arange(1, len(RandomHistory.history['accuracy'])+1))
plt.show()

plt.savefig(output_dir+'InceptionResNetV2.png')

bestEpoch = 7

RandomModel = TestNetwork(inputNet = tf.keras.applications.InceptionResNetV2,
                          **Best_Random_HP,
                          tuner_epochs = 0,
                          tuner_initial_epoch = 0,
                          tuner_bracket = 0,
                          tuner_round = 0)
RandomModel.load_weights(output_dir+f'InceptionResNetV2_RandomFinal_epoch{bestEpoch:02d}.hdf5')
RandomModel.save(output_dir+'InceptionResNetV2_RandomFinal_Model.h5')

Best_Random_HP_Updated = Best_Random_HP;
Best_Random_HP_Updated['tuner_epochs'] = bestEpoch;

f = open(output_dir+'InceptionResNetV2_RandomFinal_Model.txt', "w")
f.write(str(Best_Random_HP_Updated))
f.close()

print(RandomModel.evaluate(test_ds))