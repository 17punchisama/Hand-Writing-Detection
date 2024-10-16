import os
import tkinter as tk
from tkinter import ttk, filedialog, StringVar
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
  
# Load the saved model
model = load_model(r"C:\Users\Admin\Desktop\2nd\Linear\Hand-Writing-Detection\thai_hand_writing_detection_ver3.h5")

# Dictionary for actual class names and image paths
actual_classname = {
    "alphabet\\c0.png": "ก", "alphabet\\c1.png": "ข", "alphabet\\c2.png": "ฃ", "alphabet\\c3.png": "ค", "alphabet\\c4.png": "ฅ",
    "alphabet\\c5.png": "ฆ", "alphabet\\c6.png": "ง", "alphabet\\c7.png": "จ", "alphabet\\c8.png": "ฉ", "alphabet\\c9.png": "ช",
    "alphabet\\c10.png": "ซ", "alphabet\\c11.png": "ฌ", "alphabet\\c12.png": "ญ", "alphabet\\c13.png": "ฎ", "alphabet\\c14.png": "ฏ",
    "alphabet\\c15.png": "ฐ", "alphabet\\c16.png": "ฑ", "alphabet\\c17.png": "ฒ", "alphabet\\c18.png": "ณ", "alphabet\\c19.png": "ด",
    "alphabet\\c20.png": "ต", "alphabet\\c21.png": "ถ", "alphabet\\c22.png": "ท", "alphabet\\c23.png": "ธ", "alphabet\\c24.png": "น",
    "alphabet\\c25.png": "บ", "alphabet\\c26.png": "ป", "alphabet\\c27.png": "ผ", "alphabet\\c28.png": "ฝ", "alphabet\\c29.png": "พ",
    "alphabet\\c30.png": "ฟ", "alphabet\\c31.png": "ภ", "alphabet\\c32.png": "ม", "alphabet\\c33.png": "ย", "alphabet\\c34.png": "ร",
    "alphabet\\c35.png": "ล", "alphabet\\c36.png": "ว", "alphabet\\c37.png": "ศ", "alphabet\\c38.png": "ษ", "alphabet\\c39.png": "ส",
    "alphabet\\c40.png": "ห", "alphabet\\c41.png": "ฬ", "alphabet\\c42.png": "อ", "alphabet\\c43.png": "ฮ",
    "alphabet\\t0.png": "อ่", "alphabet\\t1.png": "อ้", "alphabet\\t2.png": "อ๊", "alphabet\\t3.png": "อ๋", "alphabet\\t4.png": "อ็",
    "alphabet\\v0.png": "ะ", "alphabet\\v1.png": "า", "alphabet\\v2.png": "อั", "alphabet\\v3.png": "อิ", "alphabet\\v4.png": "อี",
    "alphabet\\v5.png": "อึ", "alphabet\\v6.png": "อื", "alphabet\\v7.png": "อุ", "alphabet\\v8.png": "อู", "alphabet\\v9.png": "เ",
    "alphabet\\v10.png": "แ", "alphabet\\v11.png": "โ", "alphabet\\v12.png": "อำ", "alphabet\\v13.png": "ใ", "alphabet\\v14.png": "ไ",
    "alphabet\\v15.png": "ๆ", "alphabet\\v16.png": "อ์", "alphabet\\v17.png": "ฯ", "alphabet\\v18.png": "ฤ", "alphabet\\v19.png": "ฦ"
}

# Create the main window
window = tk.Tk()
window.title("Draw & Predict")
window.geometry("800x600")

# Set canvas size
WIDTH, HEIGHT = 400, 400
IMG_SIZE = 128  # Size that the model expects

# Variables to hold mouse points
last_x, last_y = None, None

# Create an image to hold the drawing
image = Image.new("RGB", (WIDTH, HEIGHT), "white")
draw = ImageDraw.Draw(image)

# Function to start drawing
def activate_paint(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

# Function to draw lines
def paint(event):
    global last_x, last_y
    if last_x is not None and last_y is not None:
        canvas.create_line((last_x, last_y, event.x, event.y), width=5, fill='black', capstyle=tk.ROUND, smooth=True)
        draw.line((last_x, last_y, event.x, event.y), fill='black', width=5)
    last_x, last_y = event.x, event.y

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill="white")
    # Clear labels
    selected_label.config(text="Selected Class: ")
    predicted_label.config(text="Predicted Class: ")
    similarity_label.config(text="Cosine Similarity: ")

# Function to save the image
def save_image():
    image.save("handwriting.png")
    print("Image saved as handwriting.png")

# Function to convert an image to a vector
def image_to_vector(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img_resized = img.resize((100, 100))  # Resize image to 100x100
    img_array = np.array(img_resized)  # Convert to numpy array
    img_vector = img_array.flatten()  # Flatten the matrix to a 1D vector
    return img_vector

# Function to find image path from character value
def get_image_path_from_value(value):
    for key, val in actual_classname.items():
        if val == value:
            return key
    return None

# Function to update selected class label
def update_selected_class(*args):
    selected_class = clicked.get()  # Get the selected class
    selected_label.config(text=f"Selected Class: {selected_class}")

# Function to predict using the model
def predict_image():
    # Convert to RGB and resize
    img = image.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # Adjust shape for the model

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    # Create class labels based on your model's output shape
    class_labels = list(actual_classname.keys())
    
    # Get the predicted class name
    predicted_class_name = actual_classname[class_labels[predicted_class_index]]
    print(f"Predicted class index: {predicted_class_index}, class name: {predicted_class_name}")

    # Update predicted label
    predicted_label.config(text=f"Predicted Class: {predicted_class_name}")

    # Calculate cosine similarity
    image2_path = get_image_path_from_value(predicted_class_name)

    if image2_path:
        image1_vector = image_to_vector('handwriting.png')  # Image to compare
        image2_vector = image_to_vector(image2_path)  # Image corresponding to the predicted character

        # Calculate Cosine similarity
        similarity = cosine_similarity([image1_vector], [image2_vector])
        similarity_label.config(text=f"Cosine Similarity: {similarity[0][0]:.4f}")
        print(f"Cosine Similarity: {similarity[0][0]:.4f}")
    else:
        predicted_label.config(text=f"Predicted Class: {predicted_class_name} (Image not found)")





# Function to load and predict image from file
def predict_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        # Predict
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        class_labels = list(actual_classname.keys())
        predicted_class_name = actual_classname[class_labels[predicted_class_index]]

        # Update predicted label
        predicted_label.config(text=f"Predicted Class: {predicted_class_name}")

        # Calculate cosine similarity with the selected class
        selected_class = clicked.get()
        image2_path = get_image_path_from_value(selected_class)

        if image2_path:
            image1_vector = image_to_vector(file_path)  # Image to compare
            image2_vector = image_to_vector(image2_path)  # Image corresponding to the selected character

            # Calculate Cosine similarity
            similarity = cosine_similarity([image1_vector], [image2_vector])

            similarity_label.config(text=f"Cosine Similarity: {similarity_label[0][0]:.4f}")
            print(f"Cosine Similarity: {similarity_label[0][0]:.4f}")
        else:
            predicted_label.config(text=f"Predicted Class: {predicted_class_name} (Image not found)")

# Create a canvas for drawing
canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg="white")
canvas.grid(row=0, column=0, columnspan=4, pady=(20, 0))

# Dropdown for selecting class
class_keys = list(actual_classname.values())
clicked = StringVar(window)
clicked.set("Select a class")  # Default value
clicked.trace("w", update_selected_class)  # Update selected class when dropdown changes

# Create Combobox for selecting class
selected_text = ttk.Combobox(window, textvariable=clicked, values=class_keys, state='readonly')
selected_text.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
selected_text.current(0)  # Set default value

# Create labels for showing selected and predicted classes
selected_label = ttk.Label(window, text="Selected Class: ", font=("Helvetica", 16))
selected_label.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

predicted_label = ttk.Label(window, text="Predicted Class: ", font=("Helvetica", 16))
predicted_label.grid(row=1, column=3, padx=10, pady=10, sticky="ew")

similarity_label = ttk.Label(window, text="Cosine Similarity: ", font=("Helvetica", 16))
similarity_label.grid(row=1, column=4, padx=10, pady=10, sticky="ew")

# Create buttons for functionality
button_frame = ttk.Frame(window)
button_frame.grid(row=2, column=0, columnspan=4, pady=10)

clear_button = ttk.Button(button_frame, text="Clear", command=clear_canvas, width=15)
clear_button.grid(row=0, column=0, padx=10)

save_button = ttk.Button(button_frame, text="Save", command=save_image, width=15)
save_button.grid(row=0, column=1, padx=10)

predict_button = ttk.Button(button_frame, text="Predict", command=predict_image, width=15)
predict_button.grid(row=0, column=2, padx=10)

file_button = ttk.Button(button_frame, text="Load & Predict from File", command=predict_from_file, width=15)
file_button.grid(row=0, column=3, padx=10)

# Bind mouse events to canvas
canvas.bind("<Button-1>", activate_paint)
canvas.bind("<B1-Motion>", paint)

# Run the main loop
window.mainloop()