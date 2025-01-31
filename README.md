# 실시간 객체 분류 (Keras & OpenCV)

## 개요
이 프로젝트는 사전 학습된 Keras 딥러닝 모델을 활용하여 웹캠을 통해 실시간으로 객체를 분류하는 프로그램입니다. 모델이 캡처된 이미지를 분석하여 해당 클래스와 신뢰도를 화면에 표시합니다.

## 사용 기술
- **Python**: 기본 프로그래밍 언어
- **Keras & TensorFlow**: 사전 학습된 딥러닝 모델을 로드하고 실행
- **OpenCV**: 실시간 영상 캡처, 이미지 전처리 및 결과 표시
- **NumPy**: 수치 연산 및 이미지 변환 처리
- **Visual Studio Code**: 코드 작성 및 편집

## 주요 기능
- 학습된 모델(`bottel&mentos.h5`)을 로드하여 객체 분류
- `labels.txt`에서 클래스 라벨을 읽어옴
- 웹캠에서 이미지를 캡처하여 예측 수행
- 이미지 전처리(크기 조정, 정규화) 후 모델 입력으로 변환
- 예측된 클래스 및 신뢰도를 화면에 표시
- `ESC` 키를 눌러 프로그램 종료 가능

## 설치 방법
Python이 설치되어 있는지 확인한 후, 필요한 라이브러리를 설치하세요:

```bash
pip install tensorflow keras opencv-python numpy
```

## 실행 방법
1. `bottel&mentos.h5` 모델 파일과 `labels.txt`를 프로젝트 폴더에 배치합니다.
2. 다음 명령어를 실행합니다:

```bash
python main.py
```

3. 웹캠이 실행되며 실시간으로 객체 분류가 진행됩니다.
4. `ESC` 키를 누르면 프로그램이 종료됩니다.

## 코드 설명
- 학습된 모델을 로드:  
  ```python
  model = load_model("bottel&mentos.h5", compile=False)
  ```
- 클래스 라벨을 읽어오기:  
  ```python
  class_names = open("labels.txt", "r").readlines()
  ```
- 웹캠에서 영상 캡처:  
  ```python
  camera = cv2.VideoCapture(0)
  ```
- 예측을 위한 이미지 전처리:  
  ```python
  image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
  image = (np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1
  ```
- 모델을 이용한 예측 수행:  
  ```python
  prediction = model.predict(image)
  ```
- 예측 결과를 화면에 표시:  
  ```python
  cv2.putText(frame, f"{class_name[2:-1]} : {str(np.round(confidence_score * 100))[:-2]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
  ```

## 향후 개선 사항
- 더 많은 학습 데이터를 활용하여 모델 정확도 향상
- 실시간 예측 성능 최적화
- 객체 추적 기능 추가로 시각적 효과 개선

## 라이선스
이 프로젝트는 오픈 소스로 제공되며 자유롭게 수정 및 개선이 가능합니다.

