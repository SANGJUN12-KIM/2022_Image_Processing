import cv2
import numpy as np
import matplotlib.pyplot as plt
from plateocr import image2text
from plateocr import ocrModelConfig
plt.style.use('dark_background')

# 1. 이미지 불러오기
img_ori = cv2.imread("images/semple_1.jpg", cv2.IMREAD_COLOR)

height, width, channel = img_ori.shape

# 2. 이미지 그레이스케일 이미지로 변경하기
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 10))
plt.imshow(gray, cmap='gray')


# 3. 노이즈 제거[가우시안(GaussianBlur) 블러] 적용 및 이미지의 이진화[(adaptiveThreshold) (검은색/흰색 구분)]
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

# 4. 윤곽선 찾기
contours, _ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# 5. 윤곽선 그리기[전체 contour를 모두 그린다(contourIdx=-1)]
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

# 6. contours_dic 리스트에 contours의 정보를 모두 저장하기 위한 리스트 선언
contours_dict = []

# 7. contour의 사각형 범위를 찾아낸다.(위치(x, y)값과 높이/너비를 저장한다 > 찾아낸 값에 대한 사각형을 그린다.
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

# 8. 찾아낸 값에 대해서 저장한다[위치(x, y) 값, 크기(높이/너비), 사각형의 중심 좌표 값]
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

# 9. 윤곽선의 크기 범위 지정
MIN_AREA = 80                           # 글자가 있는 윤곽선의 최소 넓이 지정
MIN_WIDTH, MIN_HEIGHT = 2, 8            # 글자가 있는 윤곽선의 최소 크기(가로/세로) 지정
MIN_RATIO, MAX_RATIO = 0.25, 1.0        # 글자가 있는 윤곽선의 가로/세로 비율 지정

# 10. 9번에서 정의한 값들에 해당하는 대상자들을 리스트에 저장
possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']   # 가로X세로 넓이
    ratio = d['w'] / d['h']  # 가로 대비 세로 비율

    if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)    # 대상이 되는 것들만 저장해 준다(possible_contours 리스트에 저장)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# 11. 해당하는 대상자들을 네모박스에 그린다.
for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                  thickness=2)
# 12.
MAX_DIAG_MULTIPLYER = 5      # 네모박스의 대각선 길이는 인접한 네모박스까리의 중심선 길이 비율 지정(5배 내외 지정)
MAX_ANGLE_DIFF = 12.0        # 네모박스끼리의 중심선에 대한 삼각함수 COS각의 최소값 지정(12도 미만 지정)
MAX_AREA_DIFF = 0.5          # 네모박스끼리의 면적의 차이 지정(5 미만으로 지정)
MAX_WIDTH_DIFF = 0.8         # 네모박스끼리의 너비 차이 지정(0.8 미만으로 지정)
MAX_HEIGHT_DIFF = 0.2        # 네모박스끼리의 높이 차이 기정(0.2 미만으로 지정)
MIN_N_MATCHED = 3            # 조건에 부합하는 네모박스가 몇 개인지 체크 (3개 미만이면 탈락)

def find_chars(contour_list):
    matched_result_idx = []   # 최종 인덱스 결과값을 저장해 준다.

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

             # 네모박스와 네모박스 사이(중심에서 중심까지)의 거리를 구한다.
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            # 가로의 위치가 같은 네모박스인 경우 아래, 위로 네모박스가 있는 경우
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))              # 네모박스와 네모박스 중심까지의 거리에서 해당 각을 구한다.

            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])   # 면적의 비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w']                                  # 너비의 비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h']                                 # 높이의 비율

            # 상단의 조건 비교
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:

                # 위의 조건에 맞는 네모박스만 배열에 추가해 준다.
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

# 13. 후보군의 조건이 3미만이면 번호판이 아닐 확률이 있어서 후보군에서 탈락
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

# 14. 최종 후보군에 추가한다.
        matched_result_idx.append(matched_contours_idx)

# 15. 최종 후보군이 아닌 네모박스도 한번 더 비교해 본다.
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        recursive_contour_list = find_chars(unmatched_contour)

# 16. 최종 후보가 아닌 네모박스 중에 해당되는 후보를 추가시킨다.
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# 17. 최종 대상 네모박스들을 그려본다.
for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

PLATE_WIDTH_PADDING = 1.3  # 1.3
PLATE_HEIGHT_PADDING = 1.5  # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

# 18. 최종 후보들은 가지런히 돌려 놓는다.

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  # 최종 네모박스의 삼각형의 높이를 구함
    triangle_hypotenus = np.linalg.norm(                              # 빗변의 길이를 구함
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))  # 높이/빗변 (빗변 분의 높이) 이후 각도를 구한다.

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)  # 삐뚤어진 네모를 똑바로 돌린다.

    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))              # 삐뚤어진 네모를 똑바로 돌린다.

    img_cropped = cv2.getRectSubPix(                                                                # 이미지를 자른다.
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
        0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue

    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })

    #plt.subplot(len(matched_result), 1, i + 1)
    #plt.imshow(img_cropped, cmap='gray')

longest_idx, longest_text = -1, 0
plate_chars = []

for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        ratio = w / h

        if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h

    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
cv2.imwrite('plate_area.jpg', img_result)

reader = ocrModelConfig.model(custom_model=True)
text1 = image2text.read_text_area(reader, input_file='plate_area.jpg')

print(text1)