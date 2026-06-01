# Generated from: notebook15d1812bfb (1).ipynb
# Converted at: 2026-06-01T03:28:27.691Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

print("TF Version:", tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

DATA_DIR = "/kaggle/input/cad-cardiac-mri-dataset"

for root, dirs, files in os.walk(DATA_DIR):
    level = root.replace(DATA_DIR, '').count(os.sep)
    if level < 3:
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level == 2:
            print(f'{indent}  → {len(files)} images')

DATA_DIR = "/kaggle/input/cad-cardiac-mri-dataset"

for root, dirs, files in os.walk(DATA_DIR):
    level = root.replace(DATA_DIR, '').count(os.sep)
    if level < 3:
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level == 2:
            print(f'{indent}  → {len(files)} images')

import os

# Check what's in kaggle input
for item in os.listdir('/kaggle/input'):
    print(item)

import os
for item in os.listdir('/kaggle/input'):
    print(item)

import os

# Check inside datasets folder
for root, dirs, files in os.walk('/kaggle/input/datasets'):
    level = root.replace('/kaggle/input/datasets', '').count(os.sep)
    if level < 4:
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if files:
            print(f'{indent}  → {len(files)} files, example: {files[0]}')

import os

BASE = '/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset'

for root, dirs, files in os.walk(BASE):
    level = root.replace(BASE, '').count(os.sep)
    if level < 3:
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if files:
            print(f'{indent}  → {len(files)} files, example: {files[0]}')

import os
import shutil
import random

BASE = '/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset'
OUTPUT = '/kaggle/working/flat_dataset'

# Limit per class
LIMIT = 1000  # 1000 Normal + 1000 Sick = 2000 total

random.seed(42)

for class_name in ['Normal', 'Sick']:
    os.makedirs(f'{OUTPUT}/{class_name}', exist_ok=True)
    class_path = os.path.join(BASE, class_name)
    
    # Collect all images first
    all_images = []
    for patient_dir in os.listdir(class_path):
        patient_path = os.path.join(class_path, patient_dir)
        if os.path.isdir(patient_path):
            for img_file in os.listdir(patient_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(
                        os.path.join(patient_path, img_file)
                    )
    
    # Randomly pick 1000
    selected = random.sample(all_images, min(LIMIT, len(all_images)))
    
    # Copy selected images
    for i, src in enumerate(selected):
        img_name = f'{class_name}_{i:04d}.jpg'
        dst = f'{OUTPUT}/{class_name}/{img_name}'
        shutil.copy2(src, dst)
    
    print(f'{class_name}: {len(selected)} images copied')

print("\nDone!")

import os

BASE = '/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset'

# Check what file types exist
for class_name in ['Normal', 'Sick']:
    class_path = os.path.join(BASE, class_name)
    for patient_dir in os.listdir(class_path)[:2]:  # check first 2 folders
        patient_path = os.path.join(class_path, patient_dir)
        if os.path.isdir(patient_path):
            files = os.listdir(patient_path)[:5]
            print(f'{class_name}/{patient_dir}: {files}')

import os

BASE = '/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset'

# Go one level deeper
for class_name in ['Normal', 'Sick']:
    class_path = os.path.join(BASE, class_name)
    patient_dir = os.listdir(class_path)[0]
    patient_path = os.path.join(class_path, patient_dir)
    
    series_dir = os.listdir(patient_path)[0]
    series_path = os.path.join(patient_path, series_dir)
    
    files = os.listdir(series_path)[:5]
    print(f'{class_name}/{patient_dir}/{series_dir}/')
    print(f'  Files: {files}')
    print()

import os
import shutil
import random

BASE = '/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset'
OUTPUT = '/kaggle/working/flat_dataset'

LIMIT = 1000  # 1000 per class = 2000 total
random.seed(42)

for class_name in ['Normal', 'Sick']:
    os.makedirs(f'{OUTPUT}/{class_name}', exist_ok=True)
    class_path = os.path.join(BASE, class_name)
    
    # Collect all images (3 levels deep)
    all_images = []
    for patient_dir in os.listdir(class_path):
        patient_path = os.path.join(class_path, patient_dir)
        if os.path.isdir(patient_path):
            for series_dir in os.listdir(patient_path):
                series_path = os.path.join(patient_path, series_dir)
                if os.path.isdir(series_path):
                    for img_file in os.listdir(series_path):
                        if img_file.lower().endswith('.jpg'):
                            all_images.append(
                                os.path.join(series_path, img_file)
                            )
    
    print(f'{class_name}: {len(all_images)} total images found')
    
    # Randomly pick 1000
    selected = random.sample(all_images, min(LIMIT, len(all_images)))
    
    # Copy
    for i, src in enumerate(selected):
        dst = f'{OUTPUT}/{class_name}/{class_name}_{i:04d}.jpg'
        shutil.copy2(src, dst)
    
    print(f'{class_name}: {len(selected)} images copied ✓')

print("\nDone! Ready for training.")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

OUTPUT = '/kaggle/working/flat_dataset'
IMG_SIZE = 128
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    OUTPUT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_data = datagen.flow_from_directory(
    OUTPUT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

print("Classes:", train_data.class_indices)
print("Train samples:", train_data.samples)
print("Val samples:", val_data.samples)

from tensorflow.keras import layers, models

def attention_block(x):
    channel = x.shape[-1]
    avg = layers.GlobalAveragePooling2D()(x)
    avg = layers.Dense(channel // 4, activation='relu')(avg)
    avg = layers.Dense(channel, activation='sigmoid')(avg)
    avg = layers.Reshape((1, 1, channel))(avg)
    return layers.Multiply()([x, avg])

def build_cnn(num_classes=2):
    inputs = layers.Input(shape=(128, 128, 3))

    x = layers.Conv2D(32, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)

    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)

    x = attention_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_cnn(num_classes=2)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_improved_cnn(num_classes=2):
    # Use MobileNetV2 as base (lightweight + powerful)
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze first 100 layers, train last ones
    for layer in base.layers[:100]:
        layer.trainable = False
    for layer in base.layers[100:]:
        layer.trainable = True

    inputs = layers.Input(shape=(128, 128, 3))
    x = base(inputs, training=False)
    
    # Attention
    channel = x.shape[-1]
    avg = layers.GlobalAveragePooling2D()(x)
    avg_dense = layers.Dense(channel // 4, activation='relu')(avg)
    avg_dense = layers.Dense(channel, activation='sigmoid')(avg_dense)
    avg_dense = layers.Reshape((1, 1, channel))(avg_dense)
    x = layers.Multiply()([x, avg_dense])
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_improved_cnn(num_classes=2)
model.summary()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

OUTPUT = '/kaggle/working/flat_dataset'
IMG_SIZE = 128
BATCH_SIZE = 16

# Stronger augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    OUTPUT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_data = val_datagen.flow_from_directory(
    OUTPUT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

print("Classes:", train_data.class_indices)
print("Train samples:", train_data.samples)
print("Val samples:", val_data.samples)

# Two phase training

# Phase 1: Train head only
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Phase 1: Training head ===")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
)

# Phase 2: Fine-tune whole model
print("\n=== Phase 2: Fine-tuning ===")
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
)

import matplotlib.pyplot as plt

# Combine both phases
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(acc, label='Train')
ax[0].plot(val_acc, label='Val')
ax[0].axvline(x=len(history1.history['accuracy']),
              color='red', linestyle='--', label='Fine-tune start')
ax[0].set_title('Accuracy')
ax[0].legend()

ax[1].plot(loss, label='Train')
ax[1].plot(val_loss, label='Val')
ax[1].axvline(x=len(history1.history['loss']),
              color='red', linestyle='--', label='Fine-tune start')
ax[1].set_title('Loss')
ax[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

import matplotlib.pyplot as plt

acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(acc, label='Train')
ax[0].plot(val_acc, label='Val')
ax[0].axvline(x=len(history1.history['accuracy']),
              color='red', linestyle='--', label='Fine-tune start')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend()

ax[1].plot(loss, label='Train')
ax[1].plot(val_loss, label='Val')
ax[1].axvline(x=len(history1.history['loss']),
              color='red', linestyle='--', label='Fine-tune start')
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(model.predict(val_data), axis=1)
y_true = val_data.classes
labels = list(val_data.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')
plt.title('Confusion Matrix - CAD Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

def get_gradcam(model, img_array, layer_name='Conv_1'):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output,
                 model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_class = tf.argmax(preds[0])
        loss = preds[:, pred_class]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

sample_img, sample_label = next(val_data)
heatmap = get_gradcam(model, sample_img[:1])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(sample_img[0])
ax[0].set_title('Original Cardiac MRI')
ax[0].axis('off')
ax[1].imshow(heatmap, cmap='jet')
ax[1].set_title('Grad-CAM Heatmap')
ax[1].axis('off')
plt.suptitle('Model Explainability - CAD Detection')
plt.savefig('gradcam.png', dpi=150)
plt.show()

def get_gradcam(model, img_array, layer_name='mobilenetv2_1.00_128'):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output,
                 model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_class = tf.argmax(preds[0])
        loss = preds[:, pred_class]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

sample_img, sample_label = next(val_data)
heatmap = get_gradcam(model, sample_img[:1])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(sample_img[0])
ax[0].set_title('Original Cardiac MRI')
ax[0].axis('off')
ax[1].imshow(heatmap, cmap='jet')
ax[1].set_title('Grad-CAM Heatmap')
ax[1].axis('off')
plt.suptitle('Model Explainability - CAD Detection')
plt.savefig('gradcam.png', dpi=150)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Show sample predictions
sample_img, sample_label = next(val_data)
predictions = model.predict(sample_img[:8])
labels = list(val_data.class_indices.keys())

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i in range(8):
    axes[i].imshow(sample_img[i])
    pred_class = np.argmax(predictions[i])
    true_class = np.argmax(sample_label[i])
    confidence = predictions[i][pred_class] * 100
    
    color = 'green' if pred_class == true_class else 'red'
    axes[i].set_title(
        f'True: {labels[true_class]}\n'
        f'Pred: {labels[pred_class]} ({confidence:.1f}%)',
        color=color, fontsize=9
    )
    axes[i].axis('off')

plt.suptitle('CAD Detection - Sample Predictions', fontsize=14)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Reset generator
val_data.reset()

y_pred = np.argmax(model.predict(val_data), axis=1)
y_true = val_data.classes
labels = list(val_data.class_indices.keys())

# Print report
print("="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')
plt.title('Confusion Matrix - CAD Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

OUTPUT = '/kaggle/working/flat_dataset'
IMG_SIZE = 128
BATCH_SIZE = 16

# Separate train/val/test properly
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.3  # 70% train, 30% for val+test
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    OUTPUT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42,
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    OUTPUT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False  # important!
)

print("Classes:", train_data.class_indices)
print("Train:", train_data.samples)
print("Val:", val_data.samples)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_improved_cnn(num_classes=2):
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    for layer in base.layers[:100]:
        layer.trainable = False
    for layer in base.layers[100:]:
        layer.trainable = True

    inputs = layers.Input(shape=(128, 128, 3))
    x = base(inputs, training=False)

    channel = x.shape[-1]
    avg = layers.GlobalAveragePooling2D()(x)
    avg_dense = layers.Dense(channel // 4, activation='relu')(avg)
    avg_dense = layers.Dense(channel, activation='sigmoid')(avg_dense)
    avg_dense = layers.Reshape((1, 1, channel))(avg_dense)
    x = layers.Multiply()([x, avg_dense])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_improved_cnn(num_classes=2)
print("Model built!")

# Phase 1
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Phase 1 ===")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
)

# Phase 2
print("\n=== Phase 2 ===")
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, verbose=1
        )
    ]
)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

val_data.reset()
y_pred = np.argmax(model.predict(val_data), axis=1)
y_true = val_data.classes[:len(y_pred)]
labels = list(val_data.class_indices.keys())

print("="*50)
print("FINAL CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')
plt.title('Confusion Matrix - CAD Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(acc, label='Train')
ax[0].plot(val_acc, label='Validation')
ax[0].axvline(x=len(history1.history['accuracy']),
              color='red', linestyle='--', label='Fine-tune start')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(loss, label='Train')
ax[1].plot(val_loss, label='Validation')
ax[1].axvline(x=len(history1.history['loss']),
              color='red', linestyle='--', label='Fine-tune start')
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

sample_img, sample_label = next(val_data)
predictions = model.predict(sample_img[:8])

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i in range(8):
    axes[i].imshow(sample_img[i])
    pred_class = np.argmax(predictions[i])
    true_class = np.argmax(sample_label[i])
    confidence = predictions[i][pred_class] * 100
    color = 'green' if pred_class == true_class else 'red'
    axes[i].set_title(
        f'True: {labels[true_class]}\n'
        f'Pred: {labels[pred_class]} ({confidence:.1f}%)',
        color=color, fontsize=9
    )
    axes[i].axis('off')

plt.suptitle('CAD Detection - Sample Predictions', fontsize=14)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()

def custom_data_augmentation(image):
    # 1. Random Rotation with CLEAN black borders
    if tf.random.uniform([]) > 0.5:
        angle = tf.random.uniform([], minval=-15, maxval=15).numpy()
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        # BORDER_CONSTANT with borderValue=0 prevents the noisy edge-stretching artifacts
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 2. Random Horizontal Flip
    if tf.random.uniform([]) > 0.5:
        image = cv2.flip(image, 1)

    # 3. Robust Per-Image Min-Max Normalization 
    # This prevents outlier pixels from destroying the contrast and creating gray static
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min > 0:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)

    # Ensure shape remains consistent for the model (adding channel dimension back if needed)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        
    return image.astype(np.float32)

def custom_data_augmentation(image):
    # 1. Random Rotation with CLEAN black borders
    if tf.random.uniform([]) > 0.5:
        angle = tf.random.uniform([], minval=-15, maxval=15).numpy()
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        # BORDER_CONSTANT with borderValue=0 prevents the noisy edge-stretching artifacts
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 2. Random Horizontal Flip
    if tf.random.uniform([]) > 0.5:
        image = cv2.flip(image, 1)

    # 3. Robust Per-Image Min-Max Normalization 
    # This prevents outlier pixels from destroying the contrast and creating gray static
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min > 0:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)

    # Ensure shape remains consistent for the model (adding channel dimension back if needed)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        
    return image.astype(np.float32)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set clean aesthetic style
sns.set_theme(style="whitegrid")

# =====================================================================
# PLOT 1: CONFUSION MATRIX
# =====================================================================
# Derived from your 600 total sample support, 92% Normal recall, 87% Sick recall
cm = np.array([[276, 24], 
              [39, 261]])

labels = ['Normal', 'Sick']

# Create subplots safely without using plt.figure()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, 
            annot_kws={"size": 14, "weight": "bold"}, cbar=False, ax=ax)

ax.set_title('Confusion Matrix', fontsize=16, pad=15, weight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('True Label', fontsize=12, labelpad=10)
plt.tight_layout()

# Save the figure to your directory
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()


# =====================================================================
# PLOT 2: CLASSIFICATION METRICS BAR CHART
# =====================================================================
report_data = {
    'Class': ['Normal', 'Normal', 'Normal', 'Sick', 'Sick', 'Sick'],
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Precision', 'Recall', 'F1-Score'],
    'Value': [0.88, 0.92, 0.90, 0.92, 0.87, 0.89]
}
df = pd.DataFrame(report_data)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=df, x='Metric', y='Value', hue='Class', palette='Set2', ax=ax)

ax.set_title('Classification Performance Metrics', fontsize=16, pad=15, weight='bold')
ax.set_xlabel('Metrics', fontsize=12, labelpad=10)
ax.set_ylabel('Score', fontsize=12, labelpad=10)
ax.set_ylim(0, 1.1)

# Annotate values clearly on top of each bar
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 8), 
                    textcoords='offset points', 
                    fontsize=11, weight='bold')

plt.tight_layout()

# Save the figure to your directory
plt.savefig('classification_metrics.png', dpi=300)
plt.close()

print("Evaluation images successfully generated and saved!")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set clean aesthetic style
sns.set_theme(style="whitegrid")

# =====================================================================
# PLOT 1: CONFUSION MATRIX
# =====================================================================
cm = np.array([[276, 24], 
              [39, 261]])

labels = ['Normal', 'Sick']

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, 
            annot_kws={"size": 14, "weight": "bold"}, cbar=False, ax=ax)

ax.set_title('Confusion Matrix', fontsize=16, pad=15, weight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('True Label', fontsize=12, labelpad=10)
plt.tight_layout()

# Save a copy to your directory
plt.savefig('confusion_matrix.png', dpi=300)
# Force Jupyter to display it right now
plt.show()


# =====================================================================
# PLOT 2: CLASSIFICATION METRICS BAR CHART
# =====================================================================
report_data = {
    'Class': ['Normal', 'Normal', 'Normal', 'Sick', 'Sick', 'Sick'],
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Precision', 'Recall', 'F1-Score'],
    'Value': [0.88, 0.92, 0.90, 0.92, 0.87, 0.89]
}
df = pd.DataFrame(report_data)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=df, x='Metric', y='Value', hue='Class', palette='Set2', ax=ax)

ax.set_title('Classification Performance Metrics', fontsize=16, pad=15, weight='bold')
ax.set_xlabel('Metrics', fontsize=12, labelpad=10)
ax.set_ylabel('Score', fontsize=12, labelpad=10)
ax.set_ylim(0, 1.1)

# Annotate values clearly on top of each bar
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 8), 
                    textcoords='offset points', 
                    fontsize=11, weight='bold')

plt.tight_layout()

# Save a copy to your directory
plt.savefig('classification_metrics.png', dpi=300)
# Force Jupyter to display it right now
plt.show()