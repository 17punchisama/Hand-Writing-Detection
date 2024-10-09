import os
import cv2
import pandas as pd

# โหลดข้อมูลตาราง
df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\brain\2nd\Linear\Hand-Writing-Detection\data\test\_annotations.csv')

# สร้างโฟลเดอร์สำหรับคลาสต่างๆ ถ้ายังไม่มี
for class_name in df['class'].unique():
    os.makedirs(f'data_for_test/{class_name}', exist_ok=True)

# Loop ผ่านแต่ละแถวของตาราง
for idx, row in df.iterrows():
    # โหลดรูปภาพ
    img_path = f'C:/Users/Admin/OneDrive/Desktop/brain/2nd/Linear/Hand-Writing-Detection/data/test/{row["filename"]}'
    img = cv2.imread(img_path)

    if img is not None:
        # Crop ภาพตาม bounding box
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        cropped_img = img[ymin:ymax, xmin:xmax]

        # บันทึกรูปภาพที่ crop ในโฟลเดอร์ของคลาส
        output_path = f'data_for_test/{row["class"]}/{row["filename"].split(".")[0]}_crop_{idx}.jpg'
        cv2.imwrite(output_path, cropped_img)
    else:
        print(f"Cannot load image: {img_path}")
