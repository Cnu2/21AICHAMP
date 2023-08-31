# 21 AICHAMP
- 한국전력 과제명 : 인공지능 활용 광학진단 사진 위해개소 자동적출
- 전신주의 이미지를 보고 이상판단(Status), 고장부위(Type), 고장원인(Fault)을 판단하는 모델 생성
- 최종 정확도 약 95%로 우승
- 딥러닝 라이브러리 PyTorch 사용
- 프로그래밍 언어 Python 사용

## 전처리(Pre-Processing)
- 이미지 데이터가 약 200개로 확인 결과 Data Cleansing을 적용하진 않음
- train_output.csv파일에서 Class를 숫자로 변환.
- 모델 훈련 및 추론을 위해 k = 5인 k-fold를 사용

## 탐색(EDA)

