from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = 69
# Data augmentation (ถ้าต้องการทำเพิ่ม)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Rescale pixel values to [0, 1]
    shear_range=0.2,          # Random shear
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True)     # Randomly flip images horizontally

# โหลดข้อมูลจากโฟลเดอร์แยกตามคลาส
train_generator = train_datagen.flow_from_directory(
    'Hand-Writing-Detection\data_for_training',       # Path ไปยังโฟลเดอร์หลักของ dataset
    target_size=(64, 64),     # ปรับขนาดของรูปภาพให้เป็น (64x64)
    color_mode='grayscale',   # ใช้ grayscale ถ้าเป็นข้อมูล handwriting
    batch_size=32,            # ขนาดของ batch ที่จะใช้ในการฝึก
    class_mode='categorical') # ใช้ 'categorical' สำหรับการแบ่งคลาสหลาย ๆ คลาส


# ตรวจสอบคลาสที่ ImageDataGenerator สร้างขึ้น
print(train_generator.class_indices)

# Data augmentation (ถ้าต้องการทำเพิ่ม)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Rescale pixel values to [0, 1]
    shear_range=0.2,          # Random shear
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True)     # Randomly flip images horizontally

# โหลดข้อมูลจากโฟลเดอร์แยกตามคลาส
train_generator = train_datagen.flow_from_directory(
    'Hand-Writing-Detection\data_for_training',       # Path ไปยังโฟลเดอร์หลักของ dataset
    target_size=(64, 64),     # ปรับขนาดของรูปภาพให้เป็น (64x64)
    color_mode='grayscale',   # ใช้ grayscale ถ้าเป็นข้อมูล handwriting
    batch_size=32,            # ขนาดของ batch ที่จะใช้ในการฝึก
    class_mode='categorical') # ใช้ 'categorical' สำหรับการแบ่งคลาสหลาย ๆ คลาส


# Initialize the model
model = models.Sequential()

# First Convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))  # For grayscale, (64, 64, 1) as input shape
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(128, activation='relu'))

# Output layer (for multi-class classification, use softmax)
model.add(layers.Dense(num_classes, activation='softmax'))  # num_classes is the number of classes in your dataset

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])

# Data augmentation (optional, if not already pre-processed)
train_datagen = ImageDataGenerator(rescale=1./255)

# Load your dataset (replace with your dataset directory and setup)
train_generator = train_datagen.flow_from_directory(
    'Hand-Writing-Detection\data_for_training',  # Directory containing training data
    target_size=(64, 64),  # Resize images if necessary
    color_mode='grayscale',  # Use grayscale mode for handwriting data
    batch_size=32,
    class_mode='categorical')  # Adjust based on label format (categorical for multi-class)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=100)  # Adjust steps per epoch based on dataset size

# Save the model
model.save('handwriting_cnn_model.h5')

# ตรวจสอบคลาสที่ ImageDataGenerator สร้างขึ้น
print(train_generator.class_indices)

