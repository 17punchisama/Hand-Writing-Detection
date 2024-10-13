import os
import cv2
import pandas as pd

# โหลดข้อมูลตารางจาก CSV
df = pd.read_csv(r'C:\Users\User\Documents\KMITL\forCoding\Hand_Writing_Detection\data\train\_annotations.csv')

# กำหนดโฟลเดอร์สำหรับบันทึกรูปภาพที่ครอบ
base_output_path = 'C:/Users/User/Documents/KMITL/forCoding/Hand_Writing_Detection/data_for_training'

# สร้างโฟลเดอร์สำหรับคลาสต่างๆ (ถ้ายังไม่มี)
for class_name in df['class'].unique():
    os.makedirs(os.path.join(base_output_path, class_name), exist_ok=True)

# Loop ผ่านแต่ละแถวของตาราง
for idx, row in df.iterrows():
    # กำหนด path รูปภาพต้นฉบับ
    img_path = os.path.join('C:/Users/User/Documents/KMITL/forCoding/Hand_Writing_Detection/data/train', row["filename"])

    # โหลดรูปภาพ
    img = cv2.imread(img_path)

    if img is not None:
        # ครอบรูปภาพตาม bounding box
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped_img = img[ymin:ymax, xmin:xmax]

        # สร้าง path สำหรับเซฟรูปที่ครอบ
        output_path = os.path.join(
            base_output_path,
            row["class"],
            f'{os.path.splitext(row["filename"])[0]}_crop_{idx}.jpg'
        )
        # บันทึกรูปภาพที่ครอบ
        cv2.imwrite(output_path, cropped_img)
    else:
        print(f"Cannot load image: {img_path}")
