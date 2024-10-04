import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, StringVar
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths for DL model
train_dir = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\hackTHON\Soil types'
validation_dir = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\hackTHON\Soil types'

# Define the extract_features function
def extract_features(generator, model):
    print("Extracting features...")
    features = model.predict(generator, verbose=1)
    features = features.reshape((features.shape[0], -1))  # Flatten the features
    labels = generator.classes
    print("Features extracted.")
    return features, labels

# DL: Load the EfficientNetB0 model for feature extraction
print("Loading EfficientNetB0 model...")
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("EfficientNetB0 model loaded.")

# Adding dropout and pooling layers to the base model
x = efficientnet_base.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
model = tf.keras.models.Model(inputs=efficientnet_base.input, outputs=x)

# Function to classify a new image
def classify_soil(image_path, model, feature_extractor):
    print(f"Classifying image: {image_path}")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    features = feature_extractor.predict(img_array)
    features = features.flatten().reshape(1, -1)
    prediction = model.predict(features)
    class_labels = list(train_generator.class_indices.keys())
    print(f"Classified as: {class_labels[prediction[0]]}")
    return class_labels[prediction[0]]

# Function to load and preprocess images using ImageDataGenerator
def create_data_generators(train_dir, validation_dir):
    print("Creating data generators...")
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
    )
    validation_generator = datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
    )
    print("Data generators created.")
    return train_generator, validation_generator

# DL: Create data generators
train_generator, validation_generator = create_data_generators(train_dir, validation_dir)

# DL: Extract features and train the XGBoost classifier on soil types
train_features, train_labels = extract_features(train_generator, model)
print("Training XGBoost classifier...")
xgb_classifier = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_classifier.fit(train_features, train_labels)
print("XGBoost classifier trained.")

# ML: Convert CSV to pickle if not exists
csv_file_path = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\crop\files\agriculture_data.csv'
pkl_file_path = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\crop\files\agriculture_data.pkl'

if not os.path.exists(pkl_file_path):
    if os.path.exists(csv_file_path):
        print("Converting CSV to pickle...")
        data = pd.read_csv(csv_file_path)
        data.to_pickle(pkl_file_path)
        print("CSV converted to pickle.")

# ML: Load the DataFrame from the pickle file
if os.path.exists(pkl_file_path):
    print("Loading data from pickle file...")
    data = pd.read_pickle(pkl_file_path)
    print("Data loaded.")

# ML: Preprocess data and encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for column in ['Crop', 'Season', 'State', 'Fertilizer', 'Soil_Type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
print("Categorical variables encoded.")

# ML: Train-test split
X = data[['Soil_Type']]
y_crop = data['Crop']
y_yield = data['Yield']
y_fertilizer = data['Fertilizer']
y_season = data['Season']

print("Splitting data...")
X_train, X_test, y_crop_train, y_crop_test, y_yield_train, y_yield_test, y_fertilizer_train, y_fertilizer_test, y_season_train, y_season_test = train_test_split(
    X, y_crop, y_yield, y_fertilizer, y_season, test_size=0.2, random_state=42)
print("Data split.")

# ML: Train RandomForest models for crop, yield, fertilizer, and season predictions
print("Training RandomForest models...")
rf_crop = RandomForestClassifier(n_estimators=100, random_state=42)
rf_yield = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
rf_season = RandomForestClassifier(n_estimators=100, random_state=42)

rf_crop.fit(X_train, y_crop_train)
rf_yield.fit(X_train, y_yield_train)
rf_fertilizer.fit(X_train, y_fertilizer_train)
rf_season.fit(X_train, y_season_train)
print("RandomForest models trained.")

# Function to get additional information based on predicted soil type and state
def get_additional_info(soil_type_encoded, state):
    print(f"Getting additional information for soil type: {soil_type_encoded[0, 0]} and state: {state}")
    filtered_data = data[(data['Soil_Type'] == soil_type_encoded[0, 0]) & (data['State'] == state)]
    
    # Extract relevant columns
    soil_types = label_encoders['Soil_Type'].inverse_transform(filtered_data['Soil_Type']).tolist()
    states = label_encoders['State'].inverse_transform(filtered_data['State']).tolist()
    seasons = label_encoders['Season'].inverse_transform(filtered_data['Season']).tolist()
    yields = [f"{yield_value:.2f}" for yield_value in filtered_data['Yield']]
    fertilizers = label_encoders['Fertilizer'].inverse_transform(filtered_data['Fertilizer']).tolist()
    crops = label_encoders['Crop'].inverse_transform(filtered_data['Crop']).tolist()
    

    # Determine the maximum length of columns
    max_length = max(len(soil_types), len(states), len(seasons), len(yields), len(fertilizers), len(crops))
    
    # Extend lists to match the maximum length
    soil_types.extend([None] * (max_length - len(soil_types)))
    states.extend([None] * (max_length - len(states)))
    seasons.extend([None] * (max_length - len(seasons)))
    yields.extend([None] * (max_length - len(yields)))
    fertilizers.extend([None] * (max_length - len(fertilizers)))
    crops.extend([None] * (max_length - len(crops)))

    # Create DataFrame with the specified order of columns
    info_df = pd.DataFrame({
        'Soil_Type': soil_types,
        'State': states,
        'Season': seasons,
        'Yield': yields,
        'Fertilizer': fertilizers,
        'Crops': crops
    })
    
    # Format as a single table
    info_table = tabulate(info_df, headers='keys', tablefmt='grid')
    
    print("Additional information retrieved.")
    return info_table

# Function to make predictions based on image (DL -> ML)
def make_predictions(image_path, state):
    print("Making predictions...")
    # DL: Classify soil type from image
    soil_type = classify_soil(image_path, xgb_classifier, model)
    
    # Convert soil_type to its encoded value for ML model input
    soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])
    soil_type_encoded = np.array(soil_type_encoded).reshape(-1, 1)
    
    # ML: Predict crop, yield, fertilizer, and season based on the soil type
    crop = label_encoders['Crop'].inverse_transform(rf_crop.predict(soil_type_encoded))[0]
    yield_amount = rf_yield.predict(soil_type_encoded)[0]
    fertilizer = label_encoders['Fertilizer'].inverse_transform(rf_fertilizer.predict(soil_type_encoded))[0]
    season = label_encoders['Season'].inverse_transform(rf_season.predict(soil_type_encoded))[0]
    
    # Get additional information based on the state
    additional_info = get_additional_info(soil_type_encoded, state)
    
    print("Predictions made.")
    return crop, yield_amount, fertilizer, season, additional_info

# Create the Tkinter application class
class FarmBuddyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FarmBuddy")
        self.geometry("800x600")

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Header frame
        header_frame = tk.Frame(self, bg='#f2f2f2')
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(header_frame, text="FarmBuddy", font=("Arial", 24), bg='#f2f2f2')
        title_label.pack(pady=10)

        nav_frame = tk.Frame(header_frame, bg='#f2f2f2')
        nav_frame.pack()

        home_link = tk.Label(nav_frame, text="Home", font=("Arial", 14), bg='#f2f2f2', fg='blue', cursor="hand2")
        home_link.pack(side=tk.LEFT, padx=10)
        about_link = tk.Label(nav_frame, text="About", font=("Arial", 14), bg='#f2f2f2', fg='blue', cursor="hand2")
        about_link.pack(side=tk.LEFT, padx=10)

        # Main content frame
        main_frame = tk.Frame(self)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        welcome_label = tk.Label(main_frame, text="Welcome to FarmBuddy!", font=("Arial", 18))
        welcome_label.pack(pady=20)

        # Button to select image
        self.select_image_button = tk.Button(main_frame, text="Select Soil Image", command=self.select_image)
        self.select_image_button.pack(pady=20)

        # Dropdown for state selection
        self.state_var = StringVar(self)
        self.state_var.set('Select State')  # Default value
        self.state_dropdown = tk.OptionMenu(main_frame, self.state_var, *self.get_state_options())
        self.state_dropdown.pack(pady=20)

    def get_state_options(self):
        # Extract unique states from the data for the dropdown
        return list(label_encoders['State'].classes_)

    def select_image(self):
        image_path = filedialog.askopenfilename(title='Select an image', filetypes=[('Image files', '*.jpg *.jpeg *.png')])
        if image_path:
            state = self.state_var.get()
            if state and state != 'Select State':
                state_encoded = label_encoders['State'].transform([state])[0]
                crop, yield_amount, fertilizer, season, additional_info = make_predictions(image_path, state_encoded)
                self.display_results(crop, yield_amount, fertilizer, season, additional_info)
            else:
                messagebox.showwarning("Input Error", "Please select a state from the dropdown.")

    def display_results(self, crop, yield_amount, fertilizer, season, additional_info):
        results_window = tk.Toplevel(self)
        results_window.title("Prediction Results")
        
        results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD, width=80, height=20)
        results_text.pack(expand=True, fill='both')
        
        results = f"Predicted Crop: {crop}\n"
        results += f"Predicted Yield: {yield_amount:.2f}\n"
        results += f"Recommended Fertilizer: {fertilizer}\n"
        results += f"Recommended Season: {season}\n\n"
        
        results += "Additional Information:\n"
        results += additional_info
        
        results_text.insert(tk.END, results)
        results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = FarmBuddyApp()
    app.mainloop()
