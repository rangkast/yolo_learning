import os
import json
import glob
import shutil
import random
from ultralytics import YOLO
import torch
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))

def adjust_contrast(image, alpha=1.5):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return new_image

def adjust_color(image, hue_shift=10, saturation_scale=1.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)
    hue = cv2.add(hue, hue_shift)
    saturation = cv2.convertScaleAbs(saturation, alpha=saturation_scale, beta=0)
    hsv_image = cv2.merge([hue, saturation, value])
    new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return new_image

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated

def convert_to_yolo_format(label_file, output_dir, img_width, img_height):
    with open(label_file, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    labeling_ten = []
    for item in data['images']:
        if len(item['annotations']) == 10:
            image_path = item['file']
            labeling_ten.append(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(output_dir, "{}.txt".format(base_name))
            with open(label_path, 'w') as lf:
                for ann in item['annotations']:
                    label = int(ann['label'])
                    if label == -1:
                        continue
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
                    y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
                    width = (bbox[2] - bbox[0]) / img_width
                    height = (bbox[3] - bbox[1]) / img_height
                    lf.write("{} {} {} {} {}\n".format(label, x_center, y_center, width, height))
    return labeling_ten

def prepare_cropped_dataset(image_dir, label_dir, output_dir, labeling, contrast=True, color=True, angles=[-30, -15, 15, 30]):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))

    new_width, new_height = 640, 640
    label_count = {str(i): 0 for i in range(10)}

    for img_path in image_files:
        if all(count >= 10 for count in label_count.values()):
            break

        detect_status = False
        for labeling_data in labeling:
            if labeling_data in img_path:
                detect_status = True
                break

        if detect_status:
            img = cv2.imread(img_path)
            original_height, original_width = img.shape[:2]

            # Load label file
            lbl_path = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            src_lbl_path = os.path.join(label_dir, lbl_path)

            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, 'r') as lf:
                    lines = lf.readlines()
                
                for idx, line in enumerate(lines):
                    label, x_center, y_center, width, height = map(float, line.strip().split())
                    label = str(int(label))
                    if label_count[label] >= 10:
                        continue

                    x1 = int((x_center - width / 2) * original_width)
                    y1 = int((y_center - height / 2) * original_height)
                    x2 = int((x_center + width / 2) * original_width)
                    y2 = int((y_center + height / 2) * original_height)

                    cropped_img = img[y1:y2, x1:x2]
                    cropped_img_resized = cv2.resize(cropped_img, (new_width, new_height))

                    cropped_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{idx}.jpg"
                    img_output_path = os.path.join(img_output_dir, cropped_img_name)
                    cv2.imwrite(img_output_path, cropped_img_resized)

                    new_lbl_path = os.path.join(lbl_output_dir, f"{os.path.splitext(cropped_img_name)[0]}.txt")
                    with open(new_lbl_path, 'w') as lf:
                        lf.write(f"{label} 0.5 0.5 1.0 1.0\n")

                    label_count[label] += 1

                    # Data augmentation
                    for angle in angles:
                        rotated_img = rotate_image(cropped_img_resized, angle)
                        rotated_img_name = f"{os.path.splitext(cropped_img_name)[0]}_rot{angle}.jpg"
                        cv2.imwrite(os.path.join(img_output_dir, rotated_img_name), rotated_img)
                        rotated_lbl_path = os.path.join(lbl_output_dir, f"{os.path.splitext(rotated_img_name)[0]}.txt")
                        shutil.copy(new_lbl_path, rotated_lbl_path)

                        if contrast:
                            img_contrast = adjust_contrast(rotated_img)
                            contrast_img_name = f"{os.path.splitext(rotated_img_name)[0]}_contrast.jpg"
                            cv2.imwrite(os.path.join(img_output_dir, contrast_img_name), img_contrast)
                            contrast_lbl_path = os.path.join(lbl_output_dir, f"{os.path.splitext(contrast_img_name)[0]}.txt")
                            shutil.copy(new_lbl_path, contrast_lbl_path)

                        if color:
                            img_color = adjust_color(rotated_img)
                            color_img_name = f"{os.path.splitext(rotated_img_name)[0]}_color.jpg"
                            cv2.imwrite(os.path.join(img_output_dir, color_img_name), img_color)
                            color_lbl_path = os.path.join(lbl_output_dir, f"{os.path.splitext(color_img_name)[0]}.txt")
                            shutil.copy(new_lbl_path, color_lbl_path)

def merge_datasets(original_dir, new_dir, merged_dir):
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(merged_dir, subdir), exist_ok=True)

        for file in glob.glob(os.path.join(original_dir, subdir, '*')):
            dst = os.path.join(merged_dir, subdir, os.path.basename(file))
            if os.path.abspath(file) != os.path.abspath(dst):
                shutil.copy(file, dst)

        for file in glob.glob(os.path.join(new_dir, subdir, '*')):
            dst = os.path.join(merged_dir, subdir, os.path.basename(file))
            if os.path.abspath(file) != os.path.abspath(dst):
                shutil.copy(file, dst)

def train_yolo_model(data_yaml_path, model_path='yolov8s.pt', epochs=100, batch_size=16, learning_rate=0.001):
    # 모델 로드
    model = YOLO(model_path)

    # 하이퍼파라미터 설정
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        lr0=learning_rate,
        optimizer='AdamW'  # AdamW 옵티마이저 사용
    )

    # 결과 모델 저장 경로
    result_model_dir = os.path.join(script_dir, "result_model")
    os.makedirs(result_model_dir, exist_ok=True)
    model_save_path = os.path.join(result_model_dir, "best.pt")
    model.save(model_save_path)

if __name__ == "__main__":
    img_width, img_height = 1920, 1080  
    
    # dataset_1   
    dataset_1_image_dir = os.path.join(script_dir, "images_1")
    dataset_1_label_file = os.path.join(script_dir, "labels_1.json")
    dataset_1_output_dir = os.path.join(script_dir, "yolo_dataset_1")
    label_dir = os.path.join(dataset_1_output_dir, 'labels')    
    labeling = convert_to_yolo_format(dataset_1_label_file, label_dir, img_width, img_height)
    prepare_cropped_dataset(dataset_1_image_dir, label_dir, dataset_1_output_dir, labeling)
    
    # dataset_2
    dataset_2_image_dir = os.path.join(script_dir, "images_2")
    dataset_2_label_file = os.path.join(script_dir, "labels_2.json")
    dataset_2_output_dir = os.path.join(script_dir, "yolo_dataset_2")
    label_dir = os.path.join(dataset_2_output_dir, 'labels')    
    labeling = convert_to_yolo_format(dataset_2_label_file, label_dir, img_width, img_height)
    prepare_cropped_dataset(dataset_2_image_dir, label_dir, dataset_2_output_dir, labeling)
      
    # 데이터셋 병합
    merged_output_dir = os.path.join(script_dir, "yolo_dataset_merged")    
    merge_datasets(dataset_1_output_dir, dataset_2_output_dir, merged_output_dir)
    
    data_yaml = """
    path: {}
    train: images
    val: images
    nc: 10
    names: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    """.format(merged_output_dir)
    
    with open(os.path.join(merged_output_dir, "data.yaml"), 'w') as f:
        f.write(data_yaml)
    
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    
    train_yolo_model(os.path.join(merged_output_dir, "data.yaml"), epochs=10, batch_size=2, learning_rate=0.001)
