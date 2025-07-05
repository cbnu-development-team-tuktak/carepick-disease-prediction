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

## 📊 모델 정보
- **사전학습 모델**: madatnlp/km-bert (한국어 특화 BERT)
- **모델 구조**: BERT + linear classifier (`BertForSequenceClassification`)
- **분류 대상 클래스 수**: 146개
- **학습 에폭 수**: 10 epochs
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss
- **Tokenizer**: HuggingFace BertTokenizer
- **모델 저장 형식**: PyTorch `.pt` (model_state_dict + LabelEncoder 포함)

## 💿 데이터셋 구성
- **학습 데이터**: `train_dataset.csv`
  - 감기, 폐렴 등 발병률이 높으며, 건강 삼담률이 높은 질병에 대한 증상 문장 포함
  - 스타일 변형 문장을 포함하여 일반화된 학습 유도
- **테스트 데이터**: `test_dataset_질병명.csv` (질병명 개별 저장)
  - 네이버 지식인에서 전문의가 답변한 질문 수집
  - 질병명 → `[MASK]`, 그 외 질병명 → `[DISEASE]`로 마스킹 처리

---

## 🚀 실행 방법
```bash
# 1. Google Drive 연동
from google.colab import drive
drive.mount('/content/drive')

# 2. 학습 실행
# → train_dataset.csv 경로를 지정하고 학습 epoch, learning rate 등을 설정한 후 실행

# 3. 테스트 실행
# → test_dataset_*.csv 기반 평가 수행
# → 결과는 predictions_disease_top3.csv 형태로 저장됨
```

---

## ⭐ 주요 기능
1. **질병 문장 데이터 생성**
   - 146개 질병명에 대해 다양한 문장 스타일(예: 일
3. **테스트 데이터 수집 자동화**
4. **모델 학습 및 저장**
5. **정밀 평가 도구**
