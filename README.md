# 2022_Image_Processing

---

## 설치

- python 개발환경은 3.8.5입니다.

```python
pip install https://github.com/SANGJUN12-KIM/2022_Image_Processing.git
pip install -r requirements.txt
```

## 구성

- 1_img_process.py
  작성자: 짠 반 꽁
  기능: 5번 라인의 설정된 경로인 `Number = ("images/semple_1.jpg")`이미지에 대한 전처리
  output file: `./img_processed.jpg`
- 2_plate_area_finder.py
  작성자: 오수환
  기능:9번 라인의 설정된 경로인 `img_ori = cv2.imread("images/semple_1.jpg", cv2.IMREAD_COLOR)` 이미지 중 번호판영역으로 추정되는 영역을 반환
  output file: `./plate_area.jpg`
- 3_plate_ocr.py
  작성자: 김상준
  기능: 6번 라인의 설정된 경로인 `text = image2text.read_text_area(reader, input_file='car_num_img/semple_3.jpeg')`이미지에 배치되어 있는 문자를 인식하여 반환
  output file: 없음(cmd 상 정보 반환)

