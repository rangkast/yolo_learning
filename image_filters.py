import cv2
import numpy as np

# 필터 적용 함수들 정의
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_darken(image, intensity=0.5):
    darkened_image = image * intensity
    darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8)
    return darkened_image

def apply_custom_filter(image, filters):
    for filter_func in filters:
        image = filter_func(image)
    return image

def apply_brightness(image, factor=1.0):
    brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return brightened_image

def apply_noise(image, noise_level=0.1):
    noise = np.random.randn(*image.shape) * 255 * noise_level
    noisy_image = image + noise.astype(np.uint8)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

def apply_blur(image, ksize=5):
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image


def equalizeHistogram(image):
    # 히스토그램 평활화 적용
    equalized_image = cv2.equalizeHist(image)
    # 히스토그램 평활화 결과를 BGR로 변환
    bgr_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    return bgr_image

def addFilters_on_channel(image):
    # 다양한 필터 적용
    sharp_image = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))  # 샤프닝 필터
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # 가우시안 블러
    edges = cv2.Canny(image, 100, 200)  # 엣지 디텍션
    # B, G, R 채널로 합치기
    bgr_image = cv2.merge([sharp_image, blurred_image, edges])  
    return bgr_image

def sobel_filter(image):
    # Sobel 필터 적용
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # X축 경계
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Y축 경계
    # 값을 절대값으로 변환하고 정규화
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    # B, G, R 채널로 합치기
    bgr_image = cv2.merge([image, sobelx, sobely])
    return bgr_image