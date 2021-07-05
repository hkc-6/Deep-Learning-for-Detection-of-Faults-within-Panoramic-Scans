### Evaluate models based on the 200 selected images
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import gc
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
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

### Input Parameters
batchSize = 20
AssessmentMetrics = ['accuracy', tf.keras.metrics.TrueNegatives(name = 'TN'), tf.keras.metrics.FalseNegatives(name = 'FN'), tf.keras.metrics.FalsePositives(name = 'FP')]

### Full Image
FullDs = image_dataset_from_directory('~\\Image Data\\AllImageSplit',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncFull(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncFull = keras.models.load_model('~\\ModelsAllImage_2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsAllImage_2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
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
    ResFull = keras.models.load_model('~\\ModelsAllImage_2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsAllImage_2\\ResNet50_RandomFinal_Model.txt', "r")
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
    VGGFull = keras.models.load_model('~\\ModelsAllImage_2\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsAllImage_2\\VGG16_RandomFinal_Model.txt', "r")
    VGGFullParam = eval(f.read())
    f.close()
    VGGFull.compile(
      optimizer=eval(VGGFullParam['optimiserIn']+'(lr='+str(VGGFullParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGFull

modelList1 = [(IncFull(AssessmentMetrics), FullDs, 'InceptionResNetV2 Full Image'),
              (ResFull(AssessmentMetrics), FullDs, 'ResNet50V2 Full Image'),
              (VGGFull(AssessmentMetrics), FullDs, 'VGG16 Full Image')]

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
    ### Calculate Average Recall
    _Recall = (outMetrics[2])/(outMetrics[2]+outMetrics[4])
    recall_score_list.append(_Recall)
    ### Calculate Average Precision
    _Precision = (outMetrics[2])/(outMetrics[2]+outMetrics[3])
    precision_score_list.append(_Precision)
    ### Calculate Average F1
    f1_score_list.append( (2 * _Precision * _Recall) / (_Precision + _Recall))

del IncFull, ResFull, VGGFull, modelList1, FullDs
gc.collect()



### Cropped Image
CropDs = image_dataset_from_directory('~\\Image Data\\Cropped0.001b2_2',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncCrop(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncCrop = keras.models.load_model('~\\Models0.001b2_2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2_2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
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
    ResCrop = keras.models.load_model('~\\Models0.001b2_2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2_2\\ResNet50_RandomFinal_Model.txt', "r")
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
    VGGCrop = keras.models.load_model('~\\Models0.001b2_2\\VGG16_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2_2\\VGG16_RandomFinal_Model.txt', "r")
    VGGCropParam = eval(f.read())
    f.close()
    VGGCrop.compile(
      optimizer=eval(VGGCropParam['optimiserIn']+'(lr='+str(VGGCropParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGCrop

modelList2 = [(IncCrop(AssessmentMetrics), CropDs, 'InceptionResNetV2 Cropped Image'),
              (ResCrop(AssessmentMetrics), CropDs, 'ResNet50V2 Cropped Image'),
              (VGGCrop(AssessmentMetrics), CropDs, 'VGG16 Cropped Image')]

for mod, data, name in modelList2:
    modelNames.append(name)
    ### Calculate Metrics
    outMetrics = mod.evaluate(data)
    accuracy_score_list.append(outMetrics[1])
    ### Calculate Average Recall
    _Recall = (outMetrics[2])/(outMetrics[2]+outMetrics[4])
    recall_score_list.append(_Recall)
    ### Calculate Average Precision
    _Precision = (outMetrics[2])/(outMetrics[2]+outMetrics[3])
    precision_score_list.append(_Precision)
    ### Calculate Average F1
    f1_score_list.append( (2 * _Precision * _Recall) / (_Precision + _Recall))

del IncCrop, ResCrop, VGGCrop, modelList2, CropDs
gc.collect()

### Cleaned Images
CropDs = image_dataset_from_directory('~\\Image Data\\Cropped0.001b2.Cleaned_2',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncTune(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncTune = keras.models.load_model('~\\Models0.001b2.Cleaned_2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2.Cleaned_2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
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
    ResTune = keras.models.load_model('~\\Models0.001b2.Cleaned_2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2.Cleaned_2\\ResNet50_RandomFinal_Model.txt', "r")
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
    VGGTune = keras.models.load_model('~\\Models0.001b2.Cleaned_2\\VGG16_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2.Cleaned_2\\VGG16_RandomFinal_Model.txt', "r")
    VGGTuneParam = eval(f.read())
    f.close()
    VGGTune.compile(
      optimizer=eval(VGGTuneParam['optimiserIn']+'(lr='+str(VGGTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGTune


modelList3 = [(IncTune(AssessmentMetrics), CropDs, 'InceptionResNetV2 Images Filtered'),
              (ResTune(AssessmentMetrics), CropDs, 'ResNet50V2 Images Filtered'),
              (VGGTune(AssessmentMetrics), CropDs, 'VGG16 Images Filtered')]

for mod, data, name in modelList3:
    modelNames.append(name)
    ### Calculate Metrics
    outMetrics = mod.evaluate(data)
    accuracy_score_list.append(outMetrics[1])
    ### Calculate Average Recall
    _Recall = (outMetrics[2])/(outMetrics[2]+outMetrics[4])
    recall_score_list.append(_Recall)
    ### Calculate Average Precision
    _Precision = (outMetrics[2])/(outMetrics[2]+outMetrics[3])
    precision_score_list.append(_Precision)
    ### Calculate Average F1
    f1_score_list.append( (2 * _Precision * _Recall) / (_Precision + _Recall))

del IncTune, ResTune, VGGTune, modelList3, CropDs
gc.collect()

### Append results together
Results = pd.DataFrame(modelNames, columns = ['Model'])
Results['Accuracy'] = accuracy_score_list
Results['Precision'] = precision_score_list
Results['Recall'] = recall_score_list
Results['F1'] = f1_score_list


### Export Table
filename = '~\\Final Report\\Results Table Full Data3.tex'

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

### Retrieve Hyper Parameters
### Full Images
f = open('~\\ModelsAllImage_2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncFullParam = eval(f.read())
f.close()
f = open('~\\ModelsAllImage_2\\ResNet50_RandomFinal_Model.txt', "r")
ResFullParam = eval(f.read())
f.close()
f = open('~\\ModelsAllImage_2\\VGG16_RandomFinal_Model.txt', "r")
VGGFullParam = eval(f.read())
f.close()

### Cropped Images
f = open('~\\Models0.001b2_2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncCropParam = eval(f.read())
f.close()
f = open('~\\Models0.001b2_2\\ResNet50_RandomFinal_Model.txt', "r")
ResCropParam = eval(f.read())
f.close()
f = open('~\\Models0.001b2_2\\VGG16_RandomFinal_Model.txt', "r")
VGGCropParam = eval(f.read())
f.close()

### Fine Tuned
f = open('~\\Models0.001b2.Cleaned_2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncTuneParam = eval(f.read())
f.close()
f = open('~\\Models0.001b2.Cleaned_2\\ResNet50_RandomFinal_Model.txt', "r")
ResTuneParam = eval(f.read())
f.close()
f = open('~\\Models0.001b2.Cleaned_2\\VGG16_RandomFinal_Model.txt', "r")
VGGTuneParam = eval(f.read())
f.close()

ModelList = [(IncFullParam, 'InceptionResNetV2 Full Image'),
             (ResFullParam, 'ResNet50V2 Full Image'),
             (VGGFullParam, 'VGG16 Full Image'),
             (IncCropParam, 'InceptionResNetV2 Cropped Image'),
             (ResCropParam, 'ResNet50V2 Cropped Image'),
             (VGGCropParam, 'VGG16 Cropped Image'),
             (IncTuneParam, 'InceptionResNetV2 Images Filtered'),
             (ResTuneParam, 'ResNet50V2 Images Filtered'),
             (VGGTuneParam, 'VGG16 Images Filtered')
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
paramfilename = '~\\Final Report\\Parameter Table Full Data.tex'

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


### Retrieve True/False Positive/Negatives
CropDs = image_dataset_from_directory('~\\Image Data\\Cropped0.001b2.Cleaned_2',
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

### Extract images and labels
Cropimage_batch, Croplabel_batch = CropDs2.as_numpy_iterator().next()
Cropimage_batch2 = Cropimage_batch[:20]
Croplabel_batch2 = Croplabel_batch[:20]

del CropDs, CropDs2
gc.collect()

### Extract Predictions
### Cropped
def VGGTune(AssMet):
    keras.backend.clear_session()
    gc.collect()
    VGGTune = keras.models.load_model('~\\Models0.001b2.Cleaned\\VGG16_RandomFinal_Model.h5')
    f = open('~\\Models0.001b2.Cleaned\\VGG16_RandomFinal_Model.txt', "r")
    VGGTuneParam = eval(f.read())
    f.close()
    VGGTune.compile(
      optimizer=eval(VGGTuneParam['optimiserIn']+'(lr='+str(VGGTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGTune

### Get Prediction
VGGTunePred = tf.where(VGGTune(AssessmentMetrics).predict_on_batch(Cropimage_batch2).flatten()< 0.5, 0, 1)

### Process Cropped
CropAccuracy = VGGTunePred == Croplabel_batch2
Cropimage = []
for i in Cropimage_batch2:
    Cropimage.append([i])
CropDF = pd.DataFrame(Croplabel_batch2, columns = ['Label'])
CropDF['Tag'] = ["Excellent" if i == 0 else "Shadow" for i in CropDF['Label']]
CropDF['Correct'] = CropAccuracy
CropDF['Image'] = Cropimage

del VGGTunePred, Cropimage, CropAccuracy
gc.collect()

### Extract a False Negative
FN = CropDF[(CropDF.Correct == False) & (CropDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(FN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Cleaned 2 False Negative{ImgId}.png')

### Extract a False Positive
FP = CropDF[(CropDF.Correct == False) & (CropDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(FP.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Cleaned 2 False Positive{ImgId}.png')

### Extract a True Negative
TN = CropDF[(CropDF.Correct == True) & (CropDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(TN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Cleaned 2 True Negative{ImgId}.png')

### Extract a True Positive
TP = CropDF[(CropDF.Correct == True) & (CropDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(TP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Cleaned 2 True Positive{ImgId}.png')