## SSD: Practice Single-Shot MultiBox Detector implementation in Keras
---
> This project is based on: https://github.com/pierluigiferrari/ssd_keras
#### Training details
To train the original SSD300 model on Pascal VOC:

1. Download the datasets:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  ```
2. Modify the directory to your dataset in the file `ssd300_training.py` such as:
 ```python
    VOC_2007_images_dir = '.../VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
    VOC_2007_annotations_dir = '.../VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
    VOC_2007_train_image_set_filename = '.../VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
    VOC_2007_val_image_set_filename = '.../VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
 ```
3. Run the file `ssd300_training.py`.
