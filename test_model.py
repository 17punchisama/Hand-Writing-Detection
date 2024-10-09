from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('handwriting_cnn_model.h5')
# เตรียมชุดข้อมูลทดสอบ
test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]

test_generator = test_datagen.flow_from_directory(
    r'path',  # Path ไปยังโฟลเดอร์หลักของชุดทดสอบ
    target_size=(64, 64),     # ขนาดของภาพที่ใช้ (ควรเป็นขนาดเดียวกับตอนฝึก)
    color_mode='grayscale',   # ใช้ grayscale สำหรับข้อมูลลายมือ
    batch_size=32,            # ขนาดของ batch
    class_mode='categorical',  # categorical สำหรับหลายคลาส
    shuffle=False)            # ไม่สับเปลี่ยนลำดับภาพ

# ประเมินโมเดลด้วยชุดทดสอบ
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_acc}')

# ทำการทำนายผลลัพธ์จากชุดทดสอบ
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # แปลงค่าความน่าจะเป็นเป็นคลาสที่มีค่าสูงสุด

# เปรียบเทียบผลลัพธ์ที่ทำนายกับผลลัพธ์จริง
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# แสดงผลลัพธ์ที่ทำนายและผลลัพธ์จริง
for i in range(len(predicted_classes)):
    print(f'Predicted: {class_labels[predicted_classes[i]]}, True: {class_labels[true_classes[i]]}')
