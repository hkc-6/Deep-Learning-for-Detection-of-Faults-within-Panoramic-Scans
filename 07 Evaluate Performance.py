### Assess the model performance of Stage 1

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

batchSize = 20
AssessmentMetrics = ['accuracy', tf.keras.metrics.Recall(name = 'Recall'), tf.keras.metrics.Precision(name = 'Precision')]

### Cats and Dogs
CatsDs = image_dataset_from_directory('~\\Image Data\\Image_import_test\\training_set',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(160, 160),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncCats(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncCats = keras.models.load_model('~\\ModelsCatsnDogs\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsCatsnDogs\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncCatsParam = eval(f.read())
    f.close()
    IncCats.compile(
      optimizer=eval(IncCatsParam['optimiserIn']+'(lr='+str(IncCatsParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncCats

### ResNetV2
def ResCats(AssMet):
    keras.backend.clear_session()
    gc.collect()
    ResCats = keras.models.load_model('~\\ModelsCatsnDogs\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsCatsnDogs\\ResNet50_RandomFinal_Model.txt', "r")
    ResCatsParam = eval(f.read())
    f.close()
    ResCats.compile(
      optimizer=eval(ResCatsParam['optimiserIn']+'(lr='+str(ResCatsParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return ResCats

### VGG
def VGGCats(AssMet):
    keras.backend.clear_session()
    gc.collect()
    VGGCats = keras.models.load_model('~\\ModelsCatsnDogs\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsCatsnDogs\\VGG16_RandomFinal_Model.txt', "r")
    VGGCatsParam = eval(f.read())
    f.close()
    VGGCats.compile(
      optimizer=eval(VGGCatsParam['optimiserIn']+'(lr='+str(VGGCatsParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGCats

modelList1 = [(IncCats(AssessmentMetrics), CatsDs, 'InceptionResNetV2 Verification'),
              (ResCats(AssessmentMetrics), CatsDs, 'ResNet50V2 Verification'),
              (VGGCats(AssessmentMetrics), CatsDs, 'VGG16 Verification')]

### Calculate Average Statistics
accuracy_score_list = []
recall_score_list = []
precision_score_list = []
f1_score_list = []
modelNames = []
for mod, data, name in modelList1:
    modelNames.append(name)
    ### Calculate Metrics
    outMetrics = mod.evaluate(data)
    accuracy_score_list.append(outMetrics[1])
    ### Calculate Average F1
    recall_score_list.append(outMetrics[2])
    ### Calculate Average Precision
    precision_score_list.append(outMetrics[3])
    ### Calculate Average Recall
    f1_score_list.append( (2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncCats, ResCats, VGGCats, modelList1, CatsDs
gc.collect()

### Full Image
FullDs = image_dataset_from_directory('~\\Image Data\\Image_import',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncFull(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncFull = keras.models.load_model('~\\ModelsNonCropped\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsNonCropped\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncFullParam = eval(f.read())
    f.close()
    IncFull.compile(
      optimizer=eval(IncFullParam['optimiserIn']+'(lr='+str(IncFullParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncFull

### ResNetV2
def ResFull(AssMet):
    keras.backend.clear_session()
    gc.collect()
    ResFull = keras.models.load_model('~\\ModelsNonCropped\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsNonCropped\\ResNet50_RandomFinal_Model.txt', "r")
    ResFullParam = eval(f.read())
    f.close()
    ResFull.compile(
      optimizer=eval(ResFullParam['optimiserIn']+'(lr='+str(ResFullParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return ResFull

### VGG
def VGGFull(AssMet):
    keras.backend.clear_session()
    gc.collect()
    VGGFull = keras.models.load_model('~\\ModelsNonCropped\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsNonCropped\\VGG16_RandomFinal_Model.txt', "r")
    VGGFullParam = eval(f.read())
    f.close()
    VGGFull.compile(
      optimizer=eval(VGGFullParam['optimiserIn']+'(lr='+str(VGGFullParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGFull


modelList2 = [(IncFull(AssessmentMetrics), FullDs, 'InceptionResNetV2 Full Image'),
              (ResFull(AssessmentMetrics), FullDs, 'ResNet50V2 Full Image'),
              (VGGFull(AssessmentMetrics), FullDs, 'VGG16 Full Image')]

for mod, data, name in modelList2:
    modelNames.append(name)
    ### Calculate Metrics
    outMetrics = mod.evaluate(data)
    accuracy_score_list.append(outMetrics[1])
    ### Calculate Average F1
    recall_score_list.append(outMetrics[2])
    ### Calculate Average Precision
    precision_score_list.append(outMetrics[3])
    ### Calculate Average Recall
    f1_score_list.append( (2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncFull, ResFull, VGGFull, modelList2, FullDs
gc.collect()



### Cropped Image
CropDs = image_dataset_from_directory('~\\Image Data\\CroppedImages',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncCrop(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncCrop = keras.models.load_model('~\\ModelsCropped\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncCropParam = eval(f.read())
    f.close()
    IncCrop.compile(
      optimizer=eval(IncCropParam['optimiserIn']+'(lr='+str(IncCropParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncCrop

### ResNetV2
def ResCrop(AssMet):
    keras.backend.clear_session()
    gc.collect()
    ResCrop = keras.models.load_model('~\\ModelsCropped\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped\\ResNet50_RandomFinal_Model.txt', "r")
    ResCropParam = eval(f.read())
    f.close()
    ResCrop.compile(
      optimizer=eval(ResCropParam['optimiserIn']+'(lr='+str(ResCropParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return ResCrop

### VGG
def VGGCrop(AssMet):
    keras.backend.clear_session()
    gc.collect()
    VGGCrop = keras.models.load_model('~\\ModelsCropped\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped\\VGG16_RandomFinal_Model.txt', "r")
    VGGCropParam = eval(f.read())
    f.close()
    VGGCrop.compile(
      optimizer=eval(VGGCropParam['optimiserIn']+'(lr='+str(VGGCropParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGCrop

modelList3 = [(IncCrop(AssessmentMetrics), CropDs, 'InceptionResNetV2 Cropped Image'),
              (ResCrop(AssessmentMetrics), CropDs, 'ResNet50V2 Cropped Image'),
              (VGGCrop(AssessmentMetrics), CropDs, 'VGG16 Cropped Image')]

for mod, data, name in modelList3:
    modelNames.append(name)
    ### Calculate Metrics
    outMetrics = mod.evaluate(data)
    accuracy_score_list.append(outMetrics[1])
    ### Calculate Average F1
    recall_score_list.append(outMetrics[2])
    ### Calculate Average Precision
    precision_score_list.append(outMetrics[3])
    ### Calculate Average Recall
    f1_score_list.append( (2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncCrop, ResCrop, VGGCrop, modelList3, CropDs
gc.collect()

### Fine Tuning
CropDs = image_dataset_from_directory('~\\Image Data\\CroppedImages',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncTune(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncTune = keras.models.load_model('~\\ModelsFineTune\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncTuneParam = eval(f.read())
    f.close()
    IncTune.compile(
      optimizer=eval(IncTuneParam['optimiserIn']+'(lr='+str(IncTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncTune

### ResNetV2
def ResTune(AssMet):
    keras.backend.clear_session()
    gc.collect()
    ResTune = keras.models.load_model('~\\ModelsFineTune\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune\\ResNet50_RandomFinal_Model.txt', "r")
    ResTuneParam = eval(f.read())
    f.close()
    ResTune.compile(
      optimizer=eval(ResTuneParam['optimiserIn']+'(lr='+str(ResTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return ResTune

### VGG
def VGGTune(AssMet):
    keras.backend.clear_session()
    gc.collect()
    VGGTune = keras.models.load_model('~\\ModelsFineTune\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune\\VGG16_RandomFinal_Model.txt', "r")
    VGGTuneParam = eval(f.read())
    f.close()
    VGGTune.compile(
      optimizer=eval(VGGTuneParam['optimiserIn']+'(lr='+str(VGGTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGTune


modelList4 = [(IncTune(AssessmentMetrics), CropDs, 'InceptionResNetV2 Fine Tuned'),
              (ResTune(AssessmentMetrics), CropDs, 'ResNet50V2 Fine Tuned'),
              (VGGTune(AssessmentMetrics), CropDs, 'VGG16 Fine Tuned')]

for mod, data, name in modelList4:
    modelNames.append(name)
    ### Calculate Metrics
    outMetrics = mod.evaluate(data)
    accuracy_score_list.append(outMetrics[1])
    ### Calculate Average F1
    recall_score_list.append(outMetrics[2])
    ### Calculate Average Precision
    precision_score_list.append(outMetrics[3])
    ### Calculate Average Recall
    f1_score_list.append((2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncTune, ResTune, VGGTune, modelList4, CropDs
gc.collect()


### Append results together
Results = pd.DataFrame(modelNames, columns = ['Model'])
Results['Accuracy'] = accuracy_score_list
Results['Precision'] = precision_score_list
Results['Recall'] = recall_score_list
Results['F1'] = f1_score_list


### Export Table
filename = '~\\Final Report\\Results Table.tex'
pdffile = '~\\Final Report\\Results Table.pdf'

template = r'''\documentclass[preview]{{standalone}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\begin{{document}}
\resizebox{{\textwidth}}{{!}}{{
{}
}}
\end{{document}}
'''

with open(filename, 'wb') as f:
    f.write(bytes(template.format(Results.to_latex()),'UTF-8'))

subprocess.call(['pdflatex', filename])

### Retrieve Hyper Parameters
### Cats and Dogs
f = open('~\\ModelsCatsnDogs\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncCatsParam = eval(f.read())
f.close()
f = open('~\\ModelsCatsnDogs\\ResNet50_RandomFinal_Model.txt', "r")
ResCatsParam = eval(f.read())
f.close()
f = open('~\\ModelsCatsnDogs\\VGG16_RandomFinal_Model.txt', "r")
VGGCatsParam = eval(f.read())
f.close()

### Full Images
f = open('~\\ModelsNonCropped\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncFullParam = eval(f.read())
f.close()
f = open('~\\ModelsNonCropped\\ResNet50_RandomFinal_Model.txt', "r")
ResFullParam = eval(f.read())
f.close()
f = open('~\\ModelsNonCropped\\VGG16_RandomFinal_Model.txt', "r")
VGGFullParam = eval(f.read())
f.close()

### Cropped Images
f = open('~\\ModelsCropped\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncCropParam = eval(f.read())
f.close()
f = open('~\\ModelsCropped\\ResNet50_RandomFinal_Model.txt', "r")
ResCropParam = eval(f.read())
f.close()
f = open('~\\ModelsCropped\\VGG16_RandomFinal_Model.txt', "r")
VGGCropParam = eval(f.read())
f.close()

### Fine Tuned
f = open('~\\ModelsFineTune\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncTuneParam = eval(f.read())
f.close()
f = open('~\\ModelsFineTune\\ResNet50_RandomFinal_Model.txt', "r")
ResTuneParam = eval(f.read())
f.close()
f = open('~\\ModelsFineTune\\VGG16_RandomFinal_Model.txt', "r")
VGGTuneParam = eval(f.read())
f.close()

ModelList = [(IncCatsParam, 'InceptionResNetV2 Verification'),
             (ResCatsParam, 'ResNet50V2 Verification'),
             (VGGCatsParam, 'VGG16 Verification'),
             (IncFullParam, 'InceptionResNetV2 Full Image'),
             (ResFullParam, 'ResNet50V2 Full Image'),
             (VGGFullParam, 'VGG16 Full Image'),
             (IncCropParam, 'InceptionResNetV2 Cropped Image'),
             (ResCropParam, 'ResNet50V2 Cropped Image'),
             (VGGCropParam, 'VGG16 Cropped Image'),
             (IncTuneParam, 'InceptionResNetV2 Fine Tuned'),
             (ResTuneParam, 'ResNet50V2 Fine Tuned'),
             (VGGTuneParam, 'VGG16 Fine Tuned')
             ]

modelNames2 = []
LearningRate = []
Optimiser = []
DropOut = []
Epochs = []
FineTune = []

for param, name in ModelList:
    modelNames2.append(name)
    LearningRate.append(param['learning_rateIn'])
    Optimiser.append(param['optimiserIn'])
    DropOut.append(param['DropOutIn'])
    Epochs.append(param['tuner_epochs'])
    try:
        FineTune.append(param['tuneLayersIn'])
    except:
        FineTune.append("NA")

ParamsTable = pd.DataFrame(modelNames2, columns = ['Model'])
ParamsTable['Optimiser'] = Optimiser
ParamsTable['Learning Rate'] = LearningRate
ParamsTable['Drop Out'] = DropOut
ParamsTable['Epochs'] = Epochs
ParamsTable['Fine Tune Start'] = FineTune

### Export Table
paramfilename = '~\\Final Report\\Parameter Table.tex'

template = r'''\documentclass[preview]{{standalone}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\begin{{document}}
\resizebox{{\textwidth}}{{!}}{{
{}
}}
\end{{document}}
'''

with open(paramfilename, 'wb') as f:
    f.write(bytes(template.format(ParamsTable.to_latex()),'UTF-8'))

subprocess.call(['pdflatex', paramfilename])

### Retrieve True/False Positive/Negatives
CropDs = image_dataset_from_directory('~\\Image Data\\CroppedImages',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

FullDs = image_dataset_from_directory('~\\Image Data\\Image_import',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Create one single batch
sample_size = 0
for x ,y in CropDs.unbatch():
    sample_size = sample_size + 1

CropDs2 = CropDs.unbatch().batch(sample_size)
FullDs2 = FullDs.unbatch().batch(sample_size)

### Extract images and labels
Cropimage_batch, Croplabel_batch = CropDs2.as_numpy_iterator().next()
Fullimage_batch, Fulllabel_batch = FullDs2.as_numpy_iterator().next()

del FullDs, CropDs, FullDs2, CropDs2
gc.collect()

### Extract Predictions
### Cropped
def IncCrop(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncCrop = keras.models.load_model('~\\ModelsCropped\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncCropParam = eval(f.read())
    f.close()
    IncCrop.compile(
      optimizer=eval(IncCropParam['optimiserIn']+'(lr='+str(IncCropParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncCrop

### Full
def IncFull(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncFull = keras.models.load_model('~\\ModelsNonCropped\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsNonCropped\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncFullParam = eval(f.read())
    f.close()
    IncFull.compile(
      optimizer=eval(IncFullParam['optimiserIn']+'(lr='+str(IncFullParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncFull

### Get Prediction
IncCropPred = tf.where(IncCrop(AssessmentMetrics).predict_on_batch(Cropimage_batch).flatten()< 0.5, 0, 1)
IncFullPred = tf.where(IncFull(AssessmentMetrics).predict_on_batch(Fullimage_batch).flatten()< 0.5, 0, 1)

### Process Full Image
FullAccuracy = IncFullPred == Fulllabel_batch
Fullimage = []
for i in Fullimage_batch:
    Fullimage.append([i])
FullDF = pd.DataFrame(Fulllabel_batch, columns = ['Label'])
FullDF['Tag'] = ["Excellent" if i == 0 else "Shadow" for i in FullDF['Label']]
FullDF['Correct'] = FullAccuracy
FullDF['Image'] = Fullimage

del IncFullPred, Fulllabel_batch, Fullimage, FullAccuracy
gc.collect()

matplotlib.use('Agg')

### Extract a False Negative
FN = FullDF[(FullDF.Correct == False) & (FullDF.Label == 1)]
imgId = 2
ImgTmp = np.array(FN.Image.iloc[imgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Progress Full Image False Negative{imgId}.png')

### Extract a True Negative
TN = FullDF[(FullDF.Correct == True) & (FullDF.Label == 0)]
imgId = 1
ImgTmp = np.array(TN.Image.iloc[imgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Progress Full Image True Negative{imgId}.png')

### No False Positive to Extract


### Process Cropped
CropAccuracy = IncCropPred == Croplabel_batch
Cropimage = []
for i in Cropimage_batch:
    Cropimage.append([i])
CropDF = pd.DataFrame(Croplabel_batch, columns = ['Label'])
CropDF['Tag'] = ["Excellent" if i == 0 else "Shadow" for i in CropDF['Label']]
CropDF['Correct'] = CropAccuracy
CropDF['Image'] = Cropimage

del IncCropPred, Croplabel_batch, Cropimage, CropAccuracy
gc.collect()

### Extract a False Negative
FN = CropDF[(CropDF.Correct == False) & (CropDF.Label == 1)]
ImgTmp = np.array(FN.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig('~\\Progress Cropped Image False Negative.png')

### Extract a False Positive
FP = CropDF[(CropDF.Correct == False) & (CropDF.Label == 0)]
ImgTmp = np.array(FP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig('~\\Progress Cropped Image False Positive.png')

### Extract a True Negative
TN = CropDF[(CropDF.Correct == True) & (CropDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(TN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Progress Cropped Image True Negative{ImgId}.png')

### Extract a True Positive
TP = CropDF[(CropDF.Correct == True) & (CropDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(TP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Progress Cropped Image True Positive{ImgId}.png')