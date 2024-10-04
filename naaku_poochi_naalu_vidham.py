import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pickle
import sys

# Paths
train_dir = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\hackTHON\Soil types'
validation_dir = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\hackTHON\Soil types'

# Function to load and preprocess images using ImageDataGenerator with augmentation
def create_data_generators(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
    )
    return train_generator, validation_generator

train_generator, validation_generator = create_data_generators(train_dir, validation_dir)

# Load the EfficientNetB0 model with fine-tuning
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune EfficientNetB0 model by unfreezing some of the top layers
for layer in efficientnet_base.layers[-4:]:
    layer.trainable = True

# Adding dropout and pooling layers to the base model
x = efficientnet_base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
model = Model(inputs=efficientnet_base.input, outputs=x)

# Extract features from the images using the EfficientNetB0 base model
def extract_features(generator, model):
    features = model.predict(generator, verbose=1)
    features = features.reshape((features.shape[0], -1))  # Flatten the features
    labels = generator.classes
    return features, labels

train_features, train_labels = extract_features(train_generator, model)
validation_features, validation_labels = extract_features(validation_generator, model)

# Create a pipeline with a scaler and XGBoost classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
])

# Train the pipeline
pipeline.fit(train_features, train_labels)

# Evaluate the classifier
validation_predictions = pipeline.predict(validation_features)
print(classification_report(validation_labels, validation_predictions, target_names=train_generator.class_indices.keys()))

# Save the trained model
with open('xgb_soil_classifier.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Function to classify a new image
def classify_soil(image_path, model, feature_extractor):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    features = feature_extractor.predict(img_array)
    features = features.flatten().reshape(1, -1)
    prediction = model.predict(features)
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[prediction[0]]

# Function to select an image file
def select_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title='Select an image', filetypes=[('Image files', '*.jpg *.jpeg *.png')])
    root.destroy()  # Properly close the Tkinter root window
    return image_path

# Tkinter interface for testing
def test_tkinter():
    root = tk.Tk()
    root.title("Please Wait")
    label = tk.Label(root, text="Processing...")
    label.pack(padx=20, pady=20)
    root.mainloop()

test_tkinter()

image_path = select_image()  
if image_path: 
    soil_type = classify_soil(image_path, pipeline, model)
    print(f"The soil type is: {soil_type}")
else:
    print("No image selected.")

# Exit the script properly
sys.exit()