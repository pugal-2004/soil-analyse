import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import os
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from tabulate import tabulate  # For better table formatting

# Define file paths
csv_file_path = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\crop\files\agriculture_data.csv'
pkl_file_path = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\crop\files\agriculture_data.pkl'

# Convert CSV to pickle if the pickle file does not exist
if not os.path.exists(pkl_file_path):
    if os.path.exists(csv_file_path):
        # Load the CSV file into a DataFrame
        data = pd.read_csv(csv_file_path)
        
        # Save the DataFrame to a pickle file
        data.to_pickle(pkl_file_path)
        print(f'Successfully saved the DataFrame to pickle file: {pkl_file_path}')
    else:
        print(f'The CSV file does not exist: {csv_file_path}')

# Load the DataFrame from the pickle file
if os.path.exists(pkl_file_path):
    data = pd.read_pickle(pkl_file_path)
    # Display the first few rows of the data
    print(data.head())
else:
    print(f'The pickle file does not exist: {pkl_file_path}')
    exit()  # Exit if the data cannot be loaded

# Preprocess the data
# Encode categorical variables
label_encoders = {}
for column in ['Crop', 'Season', 'State', 'Fertilizer', 'Pesticide', 'Soil_Type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target variables
X = data[['Soil_Type']]
y_crop = data['Crop']
y_yield = data['Yield']
y_fertilizer = data['Fertilizer']
y_season = data['Season']

# Train-test split
X_train, X_test, y_crop_train, y_crop_test, y_yield_train, y_yield_test, y_fertilizer_train, y_fertilizer_test, y_season_train, y_season_test = train_test_split(
    X, y_crop, y_yield, y_fertilizer, y_season, test_size=0.2, random_state=42)

# Initialize and train the RandomForest models
rf_crop = RandomForestClassifier(n_estimators=100, random_state=42)
rf_yield = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
rf_season = RandomForestClassifier(n_estimators=100, random_state=42)

rf_crop.fit(X_train, y_crop_train)
rf_yield.fit(X_train, y_yield_train)
rf_fertilizer.fit(X_train, y_fertilizer_train)
rf_season.fit(X_train, y_season_train)

# Evaluate the models
y_crop_pred = rf_crop.predict(X_test)
y_yield_pred = rf_yield.predict(X_test)
y_fertilizer_pred = rf_fertilizer.predict(X_test)
y_season_pred = rf_season.predict(X_test)

print("Crop Prediction Accuracy:", accuracy_score(y_crop_test, y_crop_pred))
print("Yield Mean Squared Error:", mean_squared_error(y_yield_test, y_yield_pred))
print("Yield Mean Absolute Error:", mean_absolute_error(y_yield_test, y_yield_pred))
print("Fertilizer Prediction Accuracy:", accuracy_score(y_fertilizer_test, y_fertilizer_pred))
print("Season Prediction Accuracy:", accuracy_score(y_season_test, y_season_pred))

# Function to make predictions based on soil type and display detailed information
def recommend_crop(soil_type):
    soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])
    soil_type_encoded = np.array(soil_type_encoded).reshape(-1, 1)  # Reshape to 2D array
    
    # Make predictions
    crop = label_encoders['Crop'].inverse_transform(rf_crop.predict(soil_type_encoded))[0]
    yield_amount = rf_yield.predict(soil_type_encoded)[0]
    fertilizer = label_encoders['Fertilizer'].inverse_transform(rf_fertilizer.predict(soil_type_encoded))[0]
    season = label_encoders['Season'].inverse_transform(rf_season.predict(soil_type_encoded))[0]
    
    # Display detailed information about the soil type
    soil_data = data[data['Soil_Type'] == label_encoders['Soil_Type'].transform([soil_type])[0]]
    
    # Decode the columns to their original values
    soil_data['Crop'] = label_encoders['Crop'].inverse_transform(soil_data['Crop'])
    soil_data['Season'] = label_encoders['Season'].inverse_transform(soil_data['Season'])
    soil_data['State'] = label_encoders['State'].inverse_transform(soil_data['State'])
    
    # Select only the desired columns
    soil_data = soil_data[['Crop', 'Season', 'State']]
    
    # Create a Tkinter window
    window = tk.Tk()
    window.title(f'Data for Soil Type: {soil_type}')
    
    # Create a scrolled text widget
    text = scrolledtext.ScrolledText(window, width=100, height=30, font=('Consolas', 10))
    text.pack(padx=10, pady=10)
    
    # Insert data into the text widget
    text.insert(tk.END, f"Recommended Crop: {crop}\n")
    text.insert(tk.END, f"Yield: {yield_amount:.2f}\n")
    text.insert(tk.END, f"Fertilizer: {fertilizer}\n")
    text.insert(tk.END, f"Season: {season}\n\n")
    text.insert(tk.END, f"Alternative\n")
    
    text.insert(tk.END, f"Data for Soil Type '{soil_type}':\n")
    text.insert(tk.END, tabulate(soil_data, headers='keys', tablefmt='grid', showindex=False))
    
    window.mainloop()
#example usage
soil_type_data = "Black"
recommend_crop(
    soil_type_data
)