import os
import random
import pandas as pd
import wandb

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from ultralytics import YOLO


train_images = '/Users/saiteja/Downloads/aquarium_pretrain/train/images'
train_labels = '/Users/saiteja/Downloads/aquarium_pretrain/train/labels'

val_images = '/Users/saiteja/Downloads/aquarium_pretrain/valid/images'
val_labels = '/Users/saiteja/Downloads/aquarium_pretrain/valid/labels'

test_images = '/Users/saiteja/Downloads/aquarium_pretrain/test/images'
test_labels = '/Users/saiteja/Downloads/aquarium_pretrain/test/labels'

yaml_path = '/Users/saiteja/Downloads/aquarium_pretrain/data.yaml'

classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
Idx2Label = {idx: label for idx, label in enumerate(classes)}
Label2Index = {label: idx for idx, label in Idx2Label.items()}
print('Index to Label Mapping:', Idx2Label)
print('Label to Index Mapping:', Label2Index)


def visualize_image_with_annotation_bboxes(image_dir, label_dir):
    image_files = sorted(os.listdir(image_dir))

    sample_image_files = random.sample(image_files, 12)

    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    for i, image_file in enumerate(sample_image_files):
        row = i // 3
        col = i % 3

        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(label_dir, image_file[:-4] + '.txt')
        f = open(label_path, 'r')

        for label in f:
            class_id, x_center, y_center, width, height = map(float, label.split())
            h, w, _ = image.shape
            x_min = int((x_center - width/2) * w)
            y_min = int((y_center - height/2) * h)
            x_max = int((x_center + width/2) * w)
            y_max = int((y_center + height/2) * h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, Idx2Label[int(class_id)], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        axs[row, col].imshow(image)
        axs[row, col].axis('off')

    plt.show()

visualize_image_with_annotation_bboxes(train_images, train_labels)

image_path = os.path.join(train_images, os.listdir(train_images)[100])
image = cv2.imread(image_path)

height, width, channels = image.shape
print('The image has dimensions {}x{} and {} channels'.format(height, width, channels))

"""## Model Training
Train mode is used for training a YOLOv8 model on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image.
"""

model = YOLO('yolov8n.pt')

torch.cuda.empty_cache()

results = model.train(
    data='/Users/saiteja/Downloads/aquarium_pretrain/data.yaml',
    epochs=2,
    imgsz=(height, width, channels),
    seed=42,
    batch=16,
    workers=4,
    name='yolov8n_custom')

df = pd.read_csv('/Users/saiteja/Downloads/runs/detect/yolov8n_custom/results.csv')
df.columns = df.columns.str.strip()

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])

axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')

plt.suptitle('Training Metrics and Loss', fontsize=24)

plt.subplots_adjust(top=0.8)

plt.tight_layout()

plt.show()


model = YOLO('/Users/saiteja/Downloads/runs/detect/yolov8n_custom/weights/best.pt')

metrics = model.val(conf=0.25, split='test')

print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")


def predict_detection(image_path):
    image = cv2.imread(image_path)

    detect_result = model(image)

    detect_image = detect_result[0].plot()

    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)

    return detect_image

image_files = sorted(os.listdir(test_images))

sample_image_files = random.sample(image_files, 12)

fig, axs = plt.subplots(4, 3, figsize=(15, 20))

for i, image_file in enumerate(sample_image_files):
    row = i // 3
    col = i % 3

    image_path = os.path.join(test_images, image_file)
    detect_image = predict_detection(image_path)

    axs[row, col].imshow(detect_image)
    axs[row, col].axis('off')

plt.show()