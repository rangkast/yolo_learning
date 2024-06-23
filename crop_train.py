# -*- coding: utf-8 -*-

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

def add_noise(image, noise_factor=0.005):
    noise = np.random.randn(*image.shape) * 255 * noise_factor
    noisy_image = image + noise.astype(np.float32)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_gaussian_blur(image, kernel_size=(3, 3), sigma=0.5):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def apply_sharpening_filter(image, alpha=0.2):
    kernel = np.array([[0, -1, 0], 
                       [-1, 4 * alpha + 1, -1], 
                       [0, -1, 0]])  # 약한 샤프니스 필터
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

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

def prepare_dataset(image_dir, label_dir, output_dir, labeling, contrast=True, color=True, noise=False, blur=False, sharpen=False):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))

    # 새로운 해상도
    hd_width, hd_height = 960, 540
    new_width, new_height = 640, 640

    for img_path in image_files:
        detect_status = False
        for labeling_data in labeling:
            if labeling_data in img_path:
                detect_status = True
                break

        if detect_status:
            img = cv2.imread(img_path)
            original_height, original_width = img.shape[:2]

            # FHD에서 HD로 변환
            hd_img = cv2.resize(img, (hd_width, hd_height))
            hd_height, hd_width = hd_img.shape[:2]

            # HD에서 640x640으로 변환
            img_resized = cv2.resize(hd_img, (new_width, new_height))
    
            img_output_path = os.path.join(img_output_dir, os.path.basename(img_path))
            cv2.imwrite(img_output_path, img_resized)

            lbl_path = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            src_lbl_path = os.path.join(label_dir, lbl_path)
            dst_lbl_path = os.path.join(lbl_output_dir, lbl_path)

            if src_lbl_path != dst_lbl_path:
                with open(src_lbl_path, 'r') as lf:
                    lines = lf.readlines()
                with open(dst_lbl_path, 'w') as lf:
                    for line in lines:
                        label, x_center, y_center, width, height = map(float, line.strip().split())
                        # FHD에서 HD로 변환
                        x_center = x_center * hd_width / original_width
                        y_center = y_center * hd_height / original_height
                        width = width * hd_width / original_width
                        height = height * hd_height / original_height
                        # HD에서 640x640으로 변환
                        x_center = x_center * new_width / hd_width
                        y_center = y_center * new_height / hd_height
                        width = width * new_width / hd_width
                        height = height * new_height / hd_height
                        lf.write(f"{label} {x_center} {y_center} {width} {height}\n")

            # 동일한 라벨 파일을 증강 이미지에 대해 생성
            if contrast:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_contrast.txt"))
            if color:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_color.txt"))
            if noise:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_noisy.txt"))
            if blur:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_blur.txt"))
            if sharpen:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_sharpen.txt"))

            # 이미지 증강
            if contrast:
                img_contrast = adjust_contrast(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_contrast.jpg"), img_contrast)
            if color:
                img_color = adjust_color(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_color.jpg"), img_color)
            if noise:
                img_noisy = add_noise(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_noisy.jpg"), img_noisy)
            if blur:
                img_blur = apply_gaussian_blur(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_blur.jpg"), img_blur)
            if sharpen:
                img_sharpen = apply_sharpening_filter(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_sharpen.jpg"), img_sharpen)

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
    prepare_dataset(dataset_1_image_dir, label_dir, dataset_1_output_dir, labeling)
    
    # dataset_2
    dataset_2_image_dir = os.path.join(script_dir, "images_2")
    dataset_2_label_file = os.path.join(script_dir, "labels_2.json")
    dataset_2_output_dir = os.path.join(script_dir, "yolo_dataset_2")
    label_dir = os.path.join(dataset_2_output_dir, 'labels')    
    labeling = convert_to_yolo_format(dataset_2_label_file, label_dir, img_width, img_height)
    prepare_dataset(dataset_2_image_dir, label_dir, dataset_2_output_dir, labeling)
      
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
    
    train_yolo_model(os.path.join(merged_output_dir, "data.yaml"), epochs=50, batch_size=2, learning_rate=0.001)
