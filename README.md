# Carepick Disease Prediction - 질병 예측 모델 학습 프로젝트

## 📄 프로젝트 소개
Carepick Disease Prediction은 사용자의 증상 입력 문장을 바탕으로, <br>
사전 학습된 BERT 기반 모델을 통해 질병을 예측하기 위한 **모델 학습 전용 프로젝트**입니다. <br>

이 프로젝트에서 학습된 모델은 Flask 기반 서버인 `carepick-self-diagnosis`에 통합되어
실시간 질병 예측 기능으로 활용됩니다.

---

## 🛠️ 기술 스택

### 📌 주요 프레임워크 및 언어
- **Python 3.10**: 데이터 처리 및 모델 학습/평가
- **Google Colab**: GPU 기반 학습 환경
- **Pandas**: CSV 처리 및 데이터프레임 연산
- **Regex / OS / CSV**: 문자열 전처리 및 파일 입출력

### 🤖 머신러닝 / 딥러닝
- **PyTorch**: 모델 구성 및 학습
- **transformers (HuggingFace)**: `madatnlp/km-bert` 기반 분류기 사용
- **scikit-learn**: `LabelEncoder`를 통한 클래스 인코딩
- **TQDM**: 학습 및 예측 진행률 시각화

### 🔍 데이터 수집
- **Request, BeautifulSoup**: 네이버 지식in에서 건강상담 글 크롤링
- **fake_useragent**: User-Agent 우회 요청

---

## 📦 
