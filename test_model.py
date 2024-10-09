import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models  # นำเข้า models
from sklearn.metrics.pairwise import cosine_similarity

# โหลดโมเดล
model = load_model('handwriting_cnn_model.h5')

# สร้าง ImageDataGenerator สำหรับการทดสอบ
test_datagen = ImageDataGenerator(rescale=1./255)

# สร้าง generator สำหรับภาพทดสอบ
test_generator = test_datagen.flow_from_directory(
    'data_for_test',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode=None,  # ไม่ต้องการ label ในตอนนี้
    shuffle=False
)

# ดึงฟีเจอร์จากโมเดล
feature_extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)  # ดึงออกมาจาก layer ก่อนสุดท้าย
features = feature_extractor.predict(test_generator)

# คำนวณ Cosine Similarity
similarity_matrix = cosine_similarity(features)

# แสดงผล Cosine Similarity
print("Cosine Similarity Matrix:")
print(similarity_matrix)

# ตัวอย่างการแสดงความคล้ายคลึงกันระหว่างภาพที่ 1 และ 2
similarity_between_1_and_2 = similarity_matrix[0][1]
print(f"Cosine Similarity between Image 1 and Image 2: {similarity_between_1_and_2}")
