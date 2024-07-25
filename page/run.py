import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from PIL import Image


# 모델 파일명 리스트(밴: model10)
model_files = [f'page/models/model{model_number}.pt' for model_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]

# 모델 로드 및 names 저장
models = []
model_names = {}

# 모델파일 이름, 모델, 클래스 저장
for model_file in model_files:
    model = YOLO(model_file)
    models.append((model_file, model)) # 모델 파일 이름도 같이 저장함
    model_names[model_file] = model.names


# 모든 모델 예측 수행, 결과 담기
def ensemble_predict(image):
    results = []
    detection_counts = []
    for model in models:
        print(f'\n{model[0]}', end="")
        result = model[1].predict(image, conf=0.5)
        results.append(result)
        # 각 모델에서 감지한 바운딩 박스의 수
        detection_counts.append(len(result[0].boxes))
    print('\ndetection_counts:')
    print(detection_counts)

    combined_results = combine_results(*results)  # final_boxes, final_confidences, final_labels
    return combined_results

# 각 결과에서 바운딩 박스, 신뢰도, 라벨 추출
def combine_results(*results):
    combined_boxes = []
    combined_confidences = []
    combined_labels = defaultdict(list)  # 같은 박스 위치에 여러 레이블을 저장하기 위한 딕셔너리

    # 각 모델의 결과에서 바운딩 박스를 추출하여 결합
    for model_index, result_list in enumerate(results):
        model_name = models[model_index][0]
        for result in result_list:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                # 클래스 번호를 클래스 이름으로 변환
                class_name = model_names[model_name][class_id]

                combined_boxes.append([x1, y1, x2, y2])
                combined_confidences.append(conf)
                combined_labels[(x1, y1, x2, y2)].append(class_name)

    if len(combined_boxes) == 0:  # combined_boxes가 비어 있는 경우를 처리합니다.
        return np.array([]), np.array([]), np.array([])

    combined_boxes = np.array(combined_boxes)
    combined_confidences = np.array(combined_confidences)

    # 박스를 그룹화하여 다수결로 라벨 결정
    final_boxes = []
    final_confidences = []
    final_labels = []

    box_groups = group_boxes_by_overlap(combined_boxes)  # 겹치는 박스 번호들끼리 그룹화한 리스트

    if len(box_groups) > 0:
        for group in box_groups:
            group_boxes = combined_boxes[group]  # 그룹의 각 번호에 해당하는 박스들의 좌표배열
            group_confidences = combined_confidences[group]  # 그룹의 각 번호에 해당하는 박스들의 신뢰도
            group_labels = [combined_labels[tuple(box)] for box in group_boxes]  # combined_labels = {(박스 좌표배열) : 라벨이름}

            # 그룹 내에서 평균 박스와 평균 신뢰도를 계산
            avg_box = np.mean(group_boxes, axis=0)
            avg_conf = np.mean(group_confidences)

            # 각 박스에 대해 다수결로 레이블 결정
            flattened_labels = [label for sublist in group_labels for label in sublist]  # 라벨이름들이 저장된 리스트
            flattened_labels = [label.lower() for label in flattened_labels] # 라벨이름 모두 소문자로 통일
            print()
            print(flattened_labels)
            most_common_label_and_count = Counter(flattened_labels).most_common(1) # 가장 빈도수가 높은 라벨 추출
            most_common_label = most_common_label_and_count[0][0]
            label_count = most_common_label_and_count[0][1]
            # print('Counter(flattened_labels):')
            # print(Counter(flattened_labels))
            # print('most_common_label_and_count:')
            # print(most_common_label_and_count)
            if label_count >= 2:
                final_boxes.append(avg_box)
                final_confidences.append(avg_conf)
                final_labels.append(most_common_label)
            else:
                return [], [], []

        final_boxes = np.array(final_boxes)
        final_confidences = np.array(final_confidences)
        final_labels = np.array(final_labels)
    else:
        return [], [], []

    return final_boxes, final_confidences, final_labels

# 박스가 겹치는 그룹을 찾는 함수(각 그룹 안에는 박스 번호들이 있음)
def group_boxes_by_overlap(boxes, iou_threshold=0.4):
    if len(boxes) == 0:  # boxes가 비어있는 경우를 처리합니다.
        return []

    distances = cdist(boxes, boxes, lambda x, y: 1 - iou(x, y))  # 두 박스간의 겹침 정도(비율)가 클수록 1에서 뺀 값이 작아지므로 거리가 작아진다.
    groups = []
    visited = set()

    for i in range(len(boxes)):  # i: 현재 박스의 인덱스
        if i in visited:  # 이미 방문된 박스 집합(visited)에 있다면 다음 반복으로 넘어감
            continue
        group = [i]  # 현재 박스 번호를 포함한 그룹 생성
        visited.add(i)  # 현재 박스 번호를 집합에 추가
        for j in range(i + 1, len(boxes)):  # 현재 박스의 다음 박스 번호부터 순회
            if j in visited:  # 이미 방문된 박스 집합(visited)에 있다면 다음 반복으로 넘어감
                continue
            if distances[i, j] < iou_threshold:  # 박스 i와 j의 거리가 iou_threshold보다 작다면, 두 박스가 많이 겹친다고 판단함
                group.append(j)  # 박스 번호 j를 group에 추가
                visited.add(j)  # 집합에도 추가
        groups.append(group)  # 그룹을 groups에 추가
    
    print('groups:')
    print(groups)

    groups = [group for group in groups if len(group) >= 3]

    return groups

# IoU (Intersection over Union) 계산 함수
def iou(box1, box2):
    x1 = max(box1[0], box2[0])  # x1 vs X1, 더 큰 값이 교집합 영역 좌상단 x좌표
    y1 = max(box1[1], box2[1])  # y1 vs Y1, 더 큰 값이 교집합 영역 좌상단 y좌표
    x2 = min(box1[2], box2[2])  # x2 vs X2, 더 작은 값이 교집합 영역 우하단 x좌표
    y2 = min(box1[3], box2[3])  # y2 vs Y2, 더 작은 값이 교집합 영역 우하단 y좌표

    intersection = max(0, x2 - x1) * max(0, y2 - y1)  # 교집합 영역 가로길이 * 교집합 영역 세로길이 = 교집합 영역의 전체 면적
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  # 박스1의 가로길이 * 세로길이 = 박스1의 전체 면적
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  # 박스2의 가로길이 * 세로길이 = 박스2의 전체 면적
    union = box1_area + box2_area - intersection  # 두 박스의 합집합 영역에서 교집합 영역을 뺀 값

    return intersection / union if union > 0 else 0  # 교집합 면적을 합집합 면적으로 나눈값


# 페이지 기본 설정
st.set_page_config(
    # 페이지 제목
    page_title='MultiCampus Enjo2조',
    # 페이지 아이콘
    page_icon='page/images/1.png'
)

# 공백
empty = st.empty()
empty.markdown('<div style="height: 200px;"></div>', unsafe_allow_html=True)

# 이미지와 제목을 한 줄에 나란히 표시하기 위해 column 두개로 나눔
col1, col2 = st.columns([2, 5])

# col1 위치에 이미지
with col1:
    st.image('page/images/1.png', width=150)

# col2 위치에 프젝 이름
with col2:
    css_title = st.markdown("""
            <style>
                .title {
                    font-size: 70px;
                    font-weight: bold;
                    color: #f481512;
                    text-shadow: 3px  0px 0 #fff;}
            </style>
            <p class=title>
                AI 요리 비서 ✨
            </p>""", unsafe_allow_html=True)

# 공백
empty1 = st.empty()
empty1.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)

# 버튼 클릭 여부를 확인하기 위한 상태 변수
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# 버튼 클릭 이벤트 처리
def start_camera():
    st.session_state.button_clicked = True

# 버튼 크기 넓히기 위해 container 생성
container = st.container()
container.button("Camera Start", on_click=start_camera, use_container_width=True)

def show_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    detected_labels = set()  # 중복 없이 탐지된 라벨을 저장할 집합(set)

    placeholder = st.empty()

    while st.session_state.button_clicked:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # 앙상블 예측 수행
        boxes, confidences, labels = ensemble_predict(frame)

        # 예측 결과를 프레임에 그리기 및 집합에 라벨 추가
        if len(boxes) > 0 and len(confidences) > 0 and len(labels) > 0:
            for box, conf, label in zip(boxes, confidences, labels):
                x1, y1, x2, y2 = map(int, box)
                label_name = label  # 이미 label은 클래스 이름입니다.

                # 바운딩 박스와 레이블 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                detected_labels.add(label_name)

        # 프레임을 BGR에서 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)

        # 이미지 업데이트
        placeholder.image(frame_image, use_column_width=True)

    # 자원 해제
    cap.release()

# 버튼이 클릭되었을 때 카메라 화면 표시 함수 호출
if st.session_state.button_clicked:
    show_camera()
