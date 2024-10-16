import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# กำหนดพารามิเตอร์เบื้องต้น
image_size = 128
batch_size = 64
num_classes = 69  # จำนวน class เป็น 69 class
epochs = 50

# 1. เตรียมข้อมูล (ไม่ทำ data augmentation)
datagen = ImageDataGenerator(rescale=1./255,)  # แค่ normalizing ข้อมูลให้อยู่ในช่วง [0, 1]

# Train generator
train_generator = datagen.flow_from_directory(
    'C:/Users/User/Documents/KMITL/forCoding/Hand_Writing_Detection/data_for_training',  # เปลี่ยนเป็น path ของโฟลเดอร์ข้อมูลการเทรน
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'  # ใช้ categorical ถ้า label เป็น one-hot encoded
)       

# Validation generator
validation_generator = datagen.flow_from_directory(
    'C:/Users/User/Documents/KMITL/forCoding/Hand_Writing_Detection/data_for_valid',  # เปลี่ยนเป็น path ของโฟลเดอร์ข้อมูล validation
    target_size=(image_size, image_size), 
    batch_size=batch_size,
    class_mode='categorical'
)

# 2. สร้างโมเดล CNN
model = models.Sequential()

# Layer 1: Convolutional Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))

# Layer 2: MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

# Layer 3: Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Layer 4: MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

# Layer 5: Convolutional Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Layer 6: MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

# Layer 7: Fully connected layer (Flatten + Dense)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))  # Fully connected layer
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with softmax for 69 class classification

# แสดงสรุปโมเดล
print(model.summary())

# 3. Compile โมเดล
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # ใช้ categorical_crossentropy เมื่อ label เป็น one-hot encoded
              metrics=['accuracy'])

# 4. Train โมเดล
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# 5. Evaluate โมเดล
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# 6. Save โมเดล (ถ้าต้องการ)
model.save('model_final_final_ver3.h5')  # บันทึกโมเดลที่เทรนแล้ว