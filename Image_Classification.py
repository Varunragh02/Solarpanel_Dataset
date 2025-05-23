import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# 1. Configuration
image_size = (224, 224)
batch_size = 16
epochs = 50
dataset_path = 'D:/Guvi/Solarpanel_Dataset/Samples'  # Update this if needed

# 2. Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 3. Training and Validation Generators
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Training class distribution:", Counter(train_generator.classes))
print("Validation class distribution:", Counter(val_generator.classes))

# 4. Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Computed class weights:", class_weights_dict)

# 5. Model Architecture
base_model = MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights='imagenet')
base_model.trainable = True  # Set to False if you want to freeze the base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

# 6. Compile Model
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Train Model
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

steps_per_epoch = int(np.ceil(train_generator.samples / batch_size))
validation_steps = int(np.ceil(val_generator.samples / batch_size))

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)
print("train_generator.class_indices", train_generator.class_indices)

# 8. Evaluation
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# 9. Save Model
model.save('solar_panel_model.h5')
print("Model saved as solar_panel_model.h5")

# 10. Plot Accuracy and Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 11. Confusion Matrix and Classification Report
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
target_names = list(val_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

# 12. Visualize Predictions
def visualize_predictions(generator, model, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        img, label = generator.next()
        pred = model.predict(img)
        pred_label = np.argmax(pred, axis=1)[0]
        true_label = np.argmax(label, axis=1)[0]

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img[0])
        plt.title(f"True: {target_names[true_label]}\nPred: {target_names[pred_label]}")
        plt.axis('off')
    plt.show()
import seaborn as sns

# 13. Plot Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

