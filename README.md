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

## ⭐ 주요 기능
1. **질병 문장 데이터 생성**
   - 프롬프트 지시 구조 설계 및 GPT 활용
   - 질병명, 문체 스타일, 조건(증상 2개 이상 포함 등)을 명시하여 실제 사용자 발화와 유사한 문장을 생성
     
2. **훈련 데이터 전처리 및 CSV 저장**
   - 생성된 문장을 파싱하여 `train_dataset.csv`로 저장
     - 항목: 질병명, 생성 문장, 스타일 번호
   - 쉼표/따옴표 포함 문장 처리 및 정규표현식을 통한 안정적 파싱
     
3. **질병 분류 모델 훈련 (PyTorch + HuggingFace)**
   - KM-BERT 기반 텍스트 분류 모델 사용
     - `madatnlp/km-bert` 모델 사용 (의료 도메인 특화)
       
   - Label Encoding 및 Train/Validation Split
     
   - 커스텀 Dataset 구성 및 학습 루프 설계
     - CrossEntropy Loss 기반 다중 클래스 분류
     - 정확도 계산 및 체크포인트 저장 기능 포함
    
4. **테스트 데이터 수집 자동화**
   - 네이버 지식iN에서 전문의 답변이 포함된 질병 문장만 수집
     - 질병 포함 여부를 나타내는 패턴 필터링 적용 (정규표현식 기반)
     - 정답 질병명을 `[MASK]`, 나머지 질병을 `[DISEASE]`로 마스킹 처리
   - 각 질병에 대해 최대 30건 수집 `test_dataset_{질병명}.csv`로 저장
     
5. **모델 학습 및 저장**
   - 예측 결과에 대해 다음 지표 계산
     - Top-K Accuracy (정답 포함 여부)
     - MRR (Mean reciprocal Rank)
     - Rank (정답 질병이 예측 결과 중 몇 번째에 있는가)
   - 평과 결과를 정확/검증 필요/부정확 중 하나로 분류
   - 최종 예측 결과를 CSV로 저장(`predictions_disease_top3.csv`)

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

## 🔧 문제 및 해결

### ❓ 단순 템플릿 기반 훈련 문장의 실제 성능 저하
#### 상황
초기에는 DB에 저장된 질병 및 증상을 기반으로, 사전에 구성한 문장 템플릿에 해당 내용을 삽입하는 방식으로 훈련 데이터를 구성하였습니다. <br>
그러나 구성된 문장이 지나치게 정형화되어 있었고, 사용자의 발화와 유사성이 낮아 실제 성능이 좋지 않은 문제가 발생하였습니다. <br>
실제 사용자 질문은 보통 증상 외에도 **발병 상황, 감정, 추측, 맥락** 등 다양한 요소를 포함하고 있기 때문입니다.
#### 해결
- [ZeroShotDataAug: Generating and Augmenting Training Data with ChatGPT](https://arxiv.org/abs/2304.14334) 논문에서 아이디어를 착안하여, **프롬프트 엔지니어링**을 통한 데이터 생성 방식으로 전환하였습니다.
- 질병별로 총 5가지 스타일의 문장을 정의하고, **스타일, 조건(증상 2개 이상 포함 등), 출력 형식(CSV 포맷)**을 명시한 프롬프트를 작성해 ChatGPT(GPT-4.o)에 전달하였습니다.
- 생성된 결과물은 사람이 직접 검수(HITL: Human-In-The-Loop)하여 **조건 위반 문장을 제거**하고 최종 학습용으로 사용하였습니다.
- 그 결과, **실제 사용자 발화**에 가까운 자연스러운 문장을 대량 확보할 수 있었고, 모델의 예측 성능 또한 유의미하게 향상되었습니다.
