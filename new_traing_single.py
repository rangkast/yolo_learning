import os
import json
import glob
import shutil
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.realpath(__file__))

def convert_to_yolo_format(label_file, output_dir, img_width, img_height):
    with open(label_file, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    labeling_ten = []
    for item in data['images']:
        if len(item['annotations']) >= 5:
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

def prepare_dataset(image_dir, label_dir, output_dir, labeling, new_width, new_height):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))

    for img_path in image_files:
        detect_status = False
        for labeling_data in labeling:
            if labeling_data in img_path:
                detect_status = True
                break

        if detect_status:
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, (new_width, new_height))
            img_output_path = os.path.join(img_output_dir, os.path.basename(img_path))
            cv2.imwrite(img_output_path, resized_img)

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
                        lf.write(f"{label} {x_center} {y_center} {width} {height}\n")              


def merge_datasets(dataset_dirs, output_dir, img_width, img_height, new_width, new_height):
    combined_image_dir = os.path.join(output_dir, 'images')
    combined_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(combined_image_dir, exist_ok=True)
    os.makedirs(combined_label_dir, exist_ok=True)

    for dataset_dir in dataset_dirs:
        image_dir = os.path.join(dataset_dir, '')
        label_file = os.path.join(dataset_dir, 'labels.json')
        print(f"Processing dataset: {dataset_dir}")
        print(f"Image directory: {image_dir}")
        print(f"Label file: {label_file}")
        if os.path.exists(image_dir) and os.path.exists(label_file):
            labeling = convert_to_yolo_format(label_file, combined_label_dir, img_width, img_height)
            prepare_dataset(image_dir, combined_label_dir, output_dir, labeling, new_width, new_height)
                
               

def split_dataset(image_dir, label_dir, output_dir, val_size=0.2, test_size=0.1, random_state=42):
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    label_files = [os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in image_files]
    # 디버깅 출력 추가
    print(f"Found {len(image_files)} image files.")
    print(f"Found {len(label_files)} label files.")
    # Train / Test split
    train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=test_size, random_state=random_state)
    
    # Train / Validation split
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_size/(1 - test_size), random_state=random_state)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)

    # Move files to respective directories
    for file in train_images:
        shutil.copy(file, os.path.join(train_dir, 'images', os.path.basename(file)))
    for file in train_labels:
        shutil.copy(file, os.path.join(train_dir, 'labels', os.path.basename(file)))
    for file in val_images:
        shutil.copy(file, os.path.join(val_dir, 'images', os.path.basename(file)))
    for file in val_labels:
        shutil.copy(file, os.path.join(val_dir, 'labels', os.path.basename(file)))
    for file in test_images:
        shutil.copy(file, os.path.join(test_dir, 'images', os.path.basename(file)))
    for file in test_labels:
        shutil.copy(file, os.path.join(test_dir, 'labels', os.path.basename(file)))
    
    return train_dir, val_dir, test_dir

def train_yolo_model(data_yaml_path, model_path='yolov8s.pt', epochs=100, batch_size=32, learning_rate=0.001, img_size=(960, 540)):
    model = YOLO(model_path)

    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        optimizer='AdamW'
    )

    # 결과 모델 저장 경로
    result_model_dir = os.path.join(script_dir, "result_model")
    os.makedirs(result_model_dir, exist_ok=True)
    model_save_path = os.path.join(result_model_dir, "best.pt")
    model.save(model_save_path)

if __name__ == "__main__":
    img_width, img_height = 1920, 1080  # 원본 이미지 크기
    new_width, new_height = 960, 544  # 새 이미지 크기
    
    # dataset 준비
    dataset_dirs = [
        os.path.join(script_dir, "./images_1"),
        os.path.join(script_dir, "./images_2"),
        os.path.join(script_dir, "../data_sets/generates_images"),
        os.path.join(script_dir, "../data_sets/Sample_1m_2nd"),
        os.path.join(script_dir, "../data_sets/Sample_1m_3rd"),
        os.path.join(script_dir, "../data_sets/Sample_1m_4th"),
        # os.path.join(script_dir, "../data_sets/hand_image"),
        # 추가 데이터셋 폴더 경로를 여기에 추가
    ]
    
    # 데이터셋 병합 및 준비
    output_dir = os.path.join(script_dir, "yolo_dataset")
    merge_datasets(dataset_dirs, output_dir, img_width, img_height, new_width, new_height)
    
    # 데이터셋 분할
    split_output_dir = os.path.join(script_dir, "yolo_dataset_split")
    train_dir, val_dir, test_dir = split_dataset(
        os.path.join(output_dir, 'images'),
        os.path.join(output_dir, 'labels'),
        split_output_dir
    )
    
    # 학습을 위한 yaml 파일 생성
    data_yaml = """
    train: {}
    val: {}
    nc: 10
    names: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    """.format(os.path.join(train_dir, 'images'), os.path.join(val_dir, 'images'))
    
    with open(os.path.join(split_output_dir, "data.yaml"), 'w') as f:
        f.write(data_yaml)
    
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    
    # 배치 크기 32로 설정
    train_yolo_model(os.path.join(split_output_dir, "data.yaml"), epochs=50, batch_size=2, learning_rate=0.001, img_size=(new_width, new_height))
