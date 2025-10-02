#!/usr/bin/env python3
"""
MTCNN과 YuNet의 눈 정렬(eye alignment) 정확도 분석
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import os

def analyze_eye_alignment(img_path, title):
    """
    정렬된 얼굴에서 양 눈의 수평 정렬 정도를 분석
    """
    if not os.path.exists(img_path):
        print(f"이미지 파일이 없습니다: {img_path}")
        return None
        
    # 이미지 로드
    img = Image.open(img_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 눈 검출을 위한 haar cascade (OpenCV 기본 제공)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 눈 검출
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(eyes) >= 2:
        # 가장 큰 두 개의 눈을 선택 (면적 기준)
        areas = [w*h for (x, y, w, h) in eyes]
        sorted_indices = np.argsort(areas)[-2:]  # 가장 큰 2개
        
        eye1 = eyes[sorted_indices[0]]
        eye2 = eyes[sorted_indices[1]]
        
        # 눈의 중심점 계산
        eye1_center = (eye1[0] + eye1[2]//2, eye1[1] + eye1[3]//2)
        eye2_center = (eye2[0] + eye2[2]//2, eye2[1] + eye2[3]//2)
        
        # 왼쪽 눈과 오른쪽 눈 구분
        if eye1_center[0] < eye2_center[0]:
            left_eye, right_eye = eye1_center, eye2_center
        else:
            left_eye, right_eye = eye2_center, eye1_center
        
        # 기울기 계산 (라디안 -> 각도)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # 눈 사이의 거리
        eye_distance = np.sqrt(dx**2 + dy**2)
        
        # 결과 이미지에 눈 위치와 기준선 그리기
        result_img = img.copy()
        draw = ImageDraw.Draw(result_img)
        
        # 눈 중심점 그리기
        draw.ellipse([left_eye[0]-3, left_eye[1]-3, left_eye[0]+3, left_eye[1]+3], 
                    fill='red', outline='darkred', width=2)
        draw.ellipse([right_eye[0]-3, right_eye[1]-3, right_eye[0]+3, right_eye[1]+3], 
                    fill='blue', outline='darkblue', width=2)
        
        # 눈을 잇는 선 그리기
        draw.line([left_eye, right_eye], fill='green', width=2)
        
        # 수평 기준선 그리기
        y_ref = (left_eye[1] + right_eye[1]) // 2
        draw.line([(0, y_ref), (img.size[0], y_ref)], fill='yellow', width=1)
        
        # 각도 표시
        mid_x = (left_eye[0] + right_eye[0]) // 2
        mid_y = (left_eye[1] + right_eye[1]) // 2
        draw.text((mid_x+10, mid_y-10), f'{angle:.1f}°', fill='white')
        
        result_img.save(f'{title.lower()}_eye_analysis.png')
        
        return {
            'title': title,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'angle_degrees': angle,
            'eye_distance': eye_distance,
            'horizontal_alignment': abs(angle) < 2.0  # 2도 이하를 수평으로 간주
        }
    else:
        print(f"{title}: 눈을 충분히 검출하지 못했습니다. (검출된 눈: {len(eyes)}개)")
        return None

def compare_alignment_quality():
    """
    MTCNN과 YuNet alignment 품질 비교
    """
    print("=== 눈 정렬(Eye Alignment) 분석 ===")
    
    # MTCNN 분석
    mtcnn_result = analyze_eye_alignment('mtcnn_aligned.png', 'MTCNN')
    
    # YuNet 분석  
    yunet_result = analyze_eye_alignment('yunet_aligned.png', 'YuNet')
    
    # 결과 출력
    if mtcnn_result:
        print(f"\n{mtcnn_result['title']} 결과:")
        print(f"  왼쪽 눈: {mtcnn_result['left_eye']}")
        print(f"  오른쪽 눈: {mtcnn_result['right_eye']}")
        print(f"  기울기: {mtcnn_result['angle_degrees']:.2f}°")
        print(f"  눈 사이 거리: {mtcnn_result['eye_distance']:.1f}px")
        print(f"  수평 정렬: {'✅' if mtcnn_result['horizontal_alignment'] else '❌'}")
    
    if yunet_result:
        print(f"\n{yunet_result['title']} 결과:")
        print(f"  왼쪽 눈: {yunet_result['left_eye']}")
        print(f"  오른쪽 눈: {yunet_result['right_eye']}")
        print(f"  기울기: {yunet_result['angle_degrees']:.2f}°")
        print(f"  눈 사이 거리: {yunet_result['eye_distance']:.1f}px")
        print(f"  수평 정렬: {'✅' if yunet_result['horizontal_alignment'] else '❌'}")
    
    # 비교 분석
    if mtcnn_result and yunet_result:
        print(f"\n=== 비교 분석 ===")
        angle_diff = abs(mtcnn_result['angle_degrees'] - yunet_result['angle_degrees'])
        distance_diff = abs(mtcnn_result['eye_distance'] - yunet_result['eye_distance'])
        
        print(f"기울기 차이: {angle_diff:.2f}°")
        print(f"눈 거리 차이: {distance_diff:.1f}px")
        
        # 더 좋은 정렬 판단
        mtcnn_better = abs(mtcnn_result['angle_degrees']) < abs(yunet_result['angle_degrees'])
        print(f"더 수평에 가까운 정렬: {'MTCNN' if mtcnn_better else 'YuNet'}")
        
        if angle_diff > 3.0:
            print("⚠️ 상당한 기울기 차이가 있습니다.")
        elif angle_diff > 1.0:
            print("⚠️ 약간의 기울기 차이가 있습니다.")
        else:
            print("✅ 비슷한 정렬 품질입니다.")

def analyze_reference_points():
    """
    MTCNN reference points의 기대값 분석
    """
    print(f"\n=== Reference Points 분석 ===")
    
    # MTCNN reference points (112x112 크기용)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_alignment'))
    from mtcnn_pytorch.src.align_trans import get_reference_facial_points
    
    ref_points_96x112 = get_reference_facial_points(
        output_size=(96, 112), default_square=False
    )
    
    ref_points_112x112 = get_reference_facial_points(
        output_size=(112, 112), default_square=True
    )
    
    print("96x112 Reference Points:")
    for i, (x, y) in enumerate(ref_points_96x112):
        labels = ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']
        print(f"  {labels[i]}: ({x:.1f}, {y:.1f})")
    
    print("112x112 Reference Points:")
    for i, (x, y) in enumerate(ref_points_112x112):
        labels = ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']
        print(f"  {labels[i]}: ({x:.1f}, {y:.1f})")
    
    # 눈의 기대 기울기 (수평이어야 함)
    eye_angle_96x112 = np.arctan2(
        ref_points_96x112[1][1] - ref_points_96x112[0][1],  # R_eye_y - L_eye_y
        ref_points_96x112[1][0] - ref_points_96x112[0][0]   # R_eye_x - L_eye_x
    ) * 180 / np.pi
    
    eye_angle_112x112 = np.arctan2(
        ref_points_112x112[1][1] - ref_points_112x112[0][1],
        ref_points_112x112[1][0] - ref_points_112x112[0][0]
    ) * 180 / np.pi
    
    print(f"\n기대 눈 기울기:")
    print(f"  96x112: {eye_angle_96x112:.2f}°")
    print(f"  112x112: {eye_angle_112x112:.2f}°")

if __name__ == "__main__":
    # Reference points 분석
    analyze_reference_points()
    
    # 실제 alignment 결과 분석
    compare_alignment_quality()