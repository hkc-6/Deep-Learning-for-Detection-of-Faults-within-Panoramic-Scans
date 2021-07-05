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
AssessmentMetrics = ['accuracy', tf.keras.metrics.Recall(name = 'Recall'), tf.keras.metrics.Precision(name = 'Precision')]

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
    ### Calculate Average F1
    recall_score_list.append(outMetrics[2])
    ### Calculate Average Precision
    precision_score_list.append(outMetrics[3])
    ### Calculate Average Recall
    f1_score_list.append( (2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncFull, ResFull, VGGFull, modelList1, FullDs
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
    IncCrop = keras.models.load_model('~\\ModelsCropped2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
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
    ResCrop = keras.models.load_model('~\\ModelsCropped2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped2\\ResNet50_RandomFinal_Model.txt', "r")
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
    VGGCrop = keras.models.load_model('~\\ModelsCropped2\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped2\\VGG16_RandomFinal_Model.txt', "r")
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
    ### Calculate Average F1
    recall_score_list.append(outMetrics[2])
    ### Calculate Average Precision
    precision_score_list.append(outMetrics[3])
    ### Calculate Average Recall
    f1_score_list.append( (2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncCrop, ResCrop, VGGCrop, modelList2, CropDs
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
    IncTune = keras.models.load_model('~\\ModelsFineTune2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
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
    ResTune = keras.models.load_model('~\\ModelsFineTune2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune2\\ResNet50_RandomFinal_Model.txt', "r")
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
    VGGTune = keras.models.load_model('~\\ModelsFineTune2\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune2\\VGG16_RandomFinal_Model.txt', "r")
    VGGTuneParam = eval(f.read())
    f.close()
    VGGTune.compile(
      optimizer=eval(VGGTuneParam['optimiserIn']+'(lr='+str(VGGTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGTune


modelList3 = [(IncTune(AssessmentMetrics), CropDs, 'InceptionResNetV2 Fine Tuned'),
              (ResTune(AssessmentMetrics), CropDs, 'ResNet50V2 Fine Tuned'),
              (VGGTune(AssessmentMetrics), CropDs, 'VGG16 Fine Tuned')]

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
    f1_score_list.append((2 * outMetrics[3] * outMetrics[2]) / (outMetrics[3] + outMetrics[2]))

del IncTune, ResTune, VGGTune, modelList3, CropDs
gc.collect()


### Object Detection
ObDtDs = image_dataset_from_directory('~\\Image Data\\CroppedImageObj',
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(270, 570),
                                      batch_size=batchSize)

### Inception ResNetV2
def IncObDt(AssMet):
    keras.backend.clear_session()
    gc.collect()
    IncObDt = keras.models.load_model('~\\ModelsCroppedImageObj2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsCroppedImageObj2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
    IncObDtParam = eval(f.read())
    f.close()
    IncObDt.compile(
      optimizer=eval(IncObDtParam['optimiserIn']+'(lr='+str(IncObDtParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return IncObDt

### ResNetV2
def ResObDt(AssMet):
    keras.backend.clear_session()
    gc.collect()
    ResObDt = keras.models.load_model('~\\ModelsCroppedImageObj2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsCroppedImageObj2\\ResNet50_RandomFinal_Model.txt', "r")
    ResObDtParam = eval(f.read())
    f.close()
    ResObDt.compile(
      optimizer=eval(ResObDtParam['optimiserIn']+'(lr='+str(ResObDtParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return ResObDt

### VGG
def VGGObDt(AssMet):
    keras.backend.clear_session()
    gc.collect()
    VGGObDt = keras.models.load_model('~\\ModelsCroppedImageObj2\\VGG16_RandomFinal_Model.h5')
    f = open('~\\ModelsCroppedImageObj2\\VGG16_RandomFinal_Model.txt', "r")
    VGGObDtParam = eval(f.read())
    f.close()
    VGGObDt.compile(
      optimizer=eval(VGGObDtParam['optimiserIn']+'(lr='+str(VGGObDtParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return VGGObDt


modelList4 = [(IncObDt(AssessmentMetrics), ObDtDs, 'InceptionResNetV2 Object Detection'),
              (ResObDt(AssessmentMetrics), ObDtDs, 'ResNet50V2 Object Detection'),
              (VGGObDt(AssessmentMetrics), ObDtDs, 'VGG16 Object Detection')]

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

del IncObDt, ResObDt, VGGObDt, modelList4, ObDtDs
gc.collect()

### Append results together
Results = pd.DataFrame(modelNames, columns = ['Model'])
Results['Accuracy'] = accuracy_score_list
Results['Precision'] = precision_score_list
Results['Recall'] = recall_score_list
Results['F1'] = f1_score_list


### Export Table
filename = '~\\Final Report\\Results Table2.tex'

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
f = open('~\\ModelsCropped2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncCropParam = eval(f.read())
f.close()
f = open('~\\ModelsCropped2\\ResNet50_RandomFinal_Model.txt', "r")
ResCropParam = eval(f.read())
f.close()
f = open('~\\ModelsCropped2\\VGG16_RandomFinal_Model.txt', "r")
VGGCropParam = eval(f.read())
f.close()

### Fine Tuned
f = open('~\\ModelsFineTune2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncTuneParam = eval(f.read())
f.close()
f = open('~\\ModelsFineTune2\\ResNet50_RandomFinal_Model.txt', "r")
ResTuneParam = eval(f.read())
f.close()
f = open('~\\ModelsFineTune2\\VGG16_RandomFinal_Model.txt', "r")
VGGTuneParam = eval(f.read())
f.close()

### Object Detection
f = open('~\\ModelsCroppedImageObj2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
IncObDtParam = eval(f.read())
f.close()
f = open('~\\ModelsCroppedImageObj2\\ResNet50_RandomFinal_Model.txt', "r")
ResObDtParam = eval(f.read())
f.close()
f = open('~\\ModelsCroppedImageObj2\\VGG16_RandomFinal_Model.txt', "r")
VGGObDtParam = eval(f.read())
f.close()

ModelList = [(IncFullParam, 'InceptionResNetV2 Full Image'),
             (ResFullParam, 'ResNet50V2 Full Image'),
             (VGGFullParam, 'VGG16 Full Image'),
             (IncCropParam, 'InceptionResNetV2 Cropped Image'),
             (ResCropParam, 'ResNet50V2 Cropped Image'),
             (VGGCropParam, 'VGG16 Cropped Image'),
             (IncTuneParam, 'InceptionResNetV2 Fine Tuned'),
             (ResTuneParam, 'ResNet50V2 Fine Tuned'),
             (VGGTuneParam, 'VGG16 Fine Tuned'),
             (IncObDtParam, 'InceptionResNetV2 Object Detection'),
             (ResObDtParam, 'ResNet50V2 Object Detection'),
             (VGGObDtParam, 'VGG16 Object Detection')
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
paramfilename = '~\\Final Report\\Parameter Table2.tex'

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
    IncCrop = keras.models.load_model('~\\ModelsCropped2\\InceptionResNetV2_RandomFinal_Model.h5')
    f = open('~\\ModelsCropped2\\InceptionResNetV2_RandomFinal_Model.txt', "r")
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

def ResTune(AssMet):
    keras.backend.clear_session()
    gc.collect()
    ResTune = keras.models.load_model('~\\ModelsFineTune2\\ResNet50_RandomFinal_Model.h5')
    f = open('~\\ModelsFineTune2\\ResNet50_RandomFinal_Model.txt', "r")
    ResTuneParam = eval(f.read())
    f.close()
    ResTune.compile(
      optimizer=eval(ResTuneParam['optimiserIn']+'(lr='+str(ResTuneParam['learning_rateIn'])+')'),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=AssMet)

    return ResTune

### Get Prediction
IncCropPred = tf.where(IncCrop(AssessmentMetrics).predict_on_batch(Cropimage_batch).flatten()< 0.5, 0, 1)
IncFullPred = tf.where(IncFull(AssessmentMetrics).predict_on_batch(Fullimage_batch).flatten()< 0.5, 0, 1)
ResTunePred = tf.where(ResTune(AssessmentMetrics).predict_on_batch(Cropimage_batch).flatten()< 0.5, 0, 1)
# IncTunePred = tf.where(IncTune(AssessmentMetrics).predict_on_batch(Cropimage_batch).flatten()< 0.5, 0, 1)

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

del IncCropPred, Cropimage, CropAccuracy
gc.collect()

### Extract a False Negative
FN = CropDF[(CropDF.Correct == False) & (CropDF.Label == 1)]
ImgTmp = np.array(FN.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig('~\\Final Cropped Image 2 False Negative.png')

### Extract a False Positive
FP = CropDF[(CropDF.Correct == False) & (CropDF.Label == 0)]
ImgTmp = np.array(FP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig('~\\Final Cropped Image 2 False Positive.png')

### Extract a True Negative
TN = CropDF[(CropDF.Correct == True) & (CropDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(TN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Cropped Image 2 True Negative{ImgId}.png')

### Extract a True Positive
TP = CropDF[(CropDF.Correct == True) & (CropDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(TP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Cropped Image 2 True Positive{ImgId}.png')



### Process Fine Tuning
TuneAccuracy = ResTunePred == Croplabel_batch
Tuneimage = []
for i in Cropimage_batch:
    Tuneimage.append([i])
TuneDF = pd.DataFrame(Croplabel_batch, columns = ['Label'])
TuneDF['Tag'] = ["Excellent" if i == 0 else "Shadow" for i in TuneDF['Label']]
TuneDF['Correct'] = TuneAccuracy
TuneDF['Image'] = Tuneimage

del ResTunePred, Tuneimage, TuneAccuracy
gc.collect()

### Extract a False Negative
FN = TuneDF[(TuneDF.Correct == False) & (TuneDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(FN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Fine Tune Image 2 False Negative{ImgId}.png')

### Extract a False Positive
FP = TuneDF[(TuneDF.Correct == False) & (TuneDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(FP.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Fine Tune Image 2 False Positive{ImgId}.png')

### Extract a True Negative
TN = TuneDF[(TuneDF.Correct == True) & (TuneDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(TN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Fine Tune Image 2 True Negative{ImgId}.png')

### Extract a True Positive
TP = TuneDF[(TuneDF.Correct == True) & (TuneDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(TP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Fine Tune Image 2 True Positive{ImgId}.png')



### Process Fine Tuning
TuneAccuracy = ResTunePred == Croplabel_batch
Tuneimage = []
for i in Cropimage_batch:
    Tuneimage.append([i])
TuneDF = pd.DataFrame(Croplabel_batch, columns = ['Label'])
TuneDF['Tag'] = ["Excellent" if i == 0 else "Shadow" for i in TuneDF['Label']]
TuneDF['Correct'] = TuneAccuracy
TuneDF['Image'] = Tuneimage

del ResTunePred, Tuneimage, TuneAccuracy
gc.collect()

### Extract a False Negative
FN = TuneDF[(TuneDF.Correct == False) & (TuneDF.Label == 1)]
ImgId = 1
ImgTmp = np.array(FN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Res Fine Tune Image 2 False Negative{ImgId}.png')

### Extract a False Positive
FP = TuneDF[(TuneDF.Correct == False) & (TuneDF.Label == 0)]
ImgId = 1
ImgTmp = np.array(FP.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Res Fine Tune Image 2 False Positive{ImgId}.png')

### Extract a True Negative
TN = TuneDF[(TuneDF.Correct == True) & (TuneDF.Label == 0)]
ImgId = 0
ImgTmp = np.array(TN.Image.iloc[ImgId])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Res Fine Tune Image 2 True Negative{ImgId}.png')

### Extract a True Positive
TP = TuneDF[(TuneDF.Correct == True) & (TuneDF.Label == 1)]
ImgId = 0
ImgTmp = np.array(TP.Image.iloc[0])/255.0
n, h, w, c = ImgTmp.shape
ImgTmp2 = ImgTmp.reshape((h, w, c))
plt.figure(figsize = (18,8))
plt.imshow(ImgTmp2, cmap=plt.cm.binary)
plt.savefig(f'~\\Final Res Fine Tune Image 2 True Positive{ImgId}.png')
