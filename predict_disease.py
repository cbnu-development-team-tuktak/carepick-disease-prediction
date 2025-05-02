# Google Drive 연동
from google.colab import drive 
drive.mount('/content/drive') # Google Drive를 '/content/drive'에 마운트

# 데이터 처리 및 모델 관련 라이브러리 import
import pandas as pd # 데이터프레임(DataFrame) 처리용 라이브러리
import torch # PyTorch 딥러닝 프레임워크
from torch.utils.data import Dataset, DataLoader # 커스텀 데이터셋과 데이터 로더 구성 도구 
from transformers import BertTokenizer, BertForSequenceClassification # BERT 토크나이저 및 분류 모델
from tqdm import tqdm # 진행 상황을 시각적으로 보여주는 프로그레스 바
import os # 파일 경로 처리용 OS 라이브러리
from torch.nn.functional import softmax # softmax 함수 (예측 확률 계산에 사용)

# ---------------------------- #
# 설정
# ---------------------------- #
model_name = "madatnlp/km-bert" # 사용할 BERT 모델명
checkpoint_path = "/content/drive/MyDrive/model_checkpoints/disease_classifier_final_18610.pt" # 학습된 모델의 체크포인트 경로
test_csv_path = "/content/drive/MyDrive/.ipynb_model_data/test_dataset_gamgi.csv" # 테스트용 CSV 파일 경로
max_len = 512 # 입력 시퀀스의 최대 길이
batch_size = 16 # 배치 시퀀스
top_n = 3 # 예측할 상위 N개의 질병 개수
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 사용할 디바이스 (GPU 사용 가능 시 CUDA)

# ---------------------------- #
# 테스트셋 로드
# ---------------------------- #
df = pd.read_csv(test_csv_path) # 테스트용 CSV 파일을 불러와 데이터프레임으로 변환
test_texts = df["text"].tolist() # 텍스트 컬럼을 리스트로 추출

# ---------------------------- #
# 토크나이저 및 데이터셋 정의
# ---------------------------- #
tokenizer = BertTokenizer.from_pretrained(model_name) # 사전 학습된 BERT 모델 토크나이저 로드

class TestDataset(
    Dataset # Python Dataset 상속
):
    def __init__(
        self, 
        texts # 입력 텍스트 리스트
    ):
        self.encodings = tokenizer(
            texts, # 입력 문장들
            truncation=True, # 최대 길이 초과 시 자르기
            padding=True, # 문장 길이를 맞추기 위해 패딩 적용
            max_length=max_len # 최대 토큰 길이 설정
        )

    def __getitem__(
        self, 
        idx # 인덱스 번호
    ):
        # 해당 인덱스의 입력 데이터를 텐서로 변환 
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

    def __len__(self):
        # 데이터셋 전체 길이 반환
        return len(self.encodings["input_ids"])

test_dataset = TestDataset(test_texts) # 테스트 문장들을 토큰화한 Dataset 객체 생성
test_loader = DataLoader(test_dataset, batch_size=batch_size) # DataLoader를 통해 배치 단위로 데이터 로드

# ---------------------------- #
# 모델 및 체크포인트 로드
# ---------------------------- #
checkpoint = torch.load(checkpoint_path, map_location=device) # 체크포인트 파일 로드
num_labels = len(checkpoint["label_encoder"].classes_) # 라벨 클래스 수 추출

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) # 사전학습된 BERT 모델 불러오기  
model.load_state_dict(checkpoint["model_state_dict"]) # 저장된 모델 가중치 적용
model.to(device) # 모델을 GPU 또는 CPU로 이동
model.eval() # 평가 모드 설정

label_encoder = checkpoint["label_encoder"] # 라벨 인코더 불러오기

# ---------------------------- #
# 예측 실행 (Top-N)
# ---------------------------- #
topn_predictions = [] # 최종 Top-N 질병 예측 결과를 저장할 리스트

with torch.no_grad(): # 그래디언트 계산 비활성화 (추론 모드)
    for batch in tqdm(test_loader, desc="예측 중"): # 테스트 데이터를 배치 단위로 반복
        batch = {k: v.to(device) for k, v in batch.items()} # 입력 데이터를 GPU 또는 CPU로 이동
        outputs = model(**batch) # 모델에 입력하여 출력(로짓) 획득
        probs = softmax(outputs.logits, dim=1) # 로짓에 소프트맥스를 적용하여 확률로 변환

        topn = torch.topk(probs, top_n, dim=1) # 각 샘플에서 확률이 높은 상위 N개 인덱스 추출
        top_indices = topn.indices.cpu().tolist() # 예측 인덱스를 리스트로 변환
        top_values = topn.values.cpu().tolist() # 예측 확률을 리스트로 변환

        for indices, scores in zip( # zip을 사용하여 예측된 질병 인덱스와 확률을 함께 순회
            top_indices, # Top-N 질병 인덱스 리스트
            top_values # Top-N 질병 확률 리스트
        ): 
            # 각 샘플의 예측 인덱스와 점수에 대해 반복
            diseases = label_encoder.inverse_transform(indices) # 인덱스를 실제 질병 이름으로 디코딩
            formatted = [f"{disease} ({score:.2f})" for disease, score in zip(diseases, scores)] # 질병명과 확률을 문자열로 변환
            topn_predictions.append(", ".join(formatted)) # 예측 결과를 하나의 문자열에 결합하여 리스트로 저장

# ---------------------------- #
# 결과 저장 (Top-N만 포함)
# ---------------------------- #
df = df[["label", "text", "url"]] # 불필요한 컬럼 제거하고 필요한 컬럼만 유지
df["top3_predicted_diseases"] = topn_predictions # 예측 결과(Top-3 질병 목록)를 새로운 컬럼으로 추가 

output_path = "predictions_gamgi_top3.csv" # 저장할 결과 CSV 파일 경로 지정
df.to_csv(output_path, index=False, encoding='utf-8-sig') # cSV 파일로 저장 (UTF-8-SIG 인코딩으로 한글 깨짐 방지)

print(f"✅ 예측 완료: {output_path} 파일 저장됨 (Top-{top_n} 결과만 포함)") # 완료 메시지 출력
