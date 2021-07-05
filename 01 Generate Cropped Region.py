### Apply the manually drawn annotations onto the 200 select image
### Requires images and annotations to be set up according to Yet-Another-EfficentDet

import os
import sys
import cv2
### define path to where the efficientdet codes are located
sys.path.append(os.path.abspath("~\\Yet-Another-EfficientDet-Pytorch\\efficientdet"))
from dataset import CocoDataset

### Define location of the dataset to apply the cropping
ds = CocoDataset(root_dir='~\\Yet-Another-EfficientDet-Pytorch\\datasets\\Tongue_Test', set='Validation_Exc')
for i in range(len(ds)):
  Bound = ds.load_annotations(i)[0]
  image = ds.load_image(i)
  crop = image[int(Bound[1]):int(Bound[3]), int(Bound[0]):int(Bound[2])]
  crop_adj = cv2.cvtColor(crop * 225., cv2.COLOR_BGR2RGB)
  ### Define the output location
  cv2.imwrite("~\\Image Data\\CroppedImages\\Excellent\\{k1}".format(k1 = ds.coco.loadImgs(ds.image_ids[i])[0]['file_name']), crop_adj)

### Define location of the dataset to apply the cropping
ds2 = CocoDataset(root_dir='~\\Yet-Another-EfficientDet-Pytorch\\datasets\\Tongue_Test', set='Training_Shad')
for i in range(len(ds2)):
  Bound = ds2.load_annotations(i)[0]
  image = ds2.load_image(i)
  crop = image[int(Bound[1]):int(Bound[3]), int(Bound[0]):int(Bound[2])]
  crop_adj = cv2.cvtColor(crop * 225., cv2.COLOR_BGR2RGB)
  ### Define the output location
  cv2.imwrite("~\\Image Data\\CroppedImages\\Shadow\\{k1}".format(k1 = ds2.coco.loadImgs(ds2.image_ids[i])[0]['file_name']), crop_adj)
