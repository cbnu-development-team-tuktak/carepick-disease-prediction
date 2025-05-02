# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 데이터 처리 관련 import
import pandas as pd # 데이터프레임 처리용

# PyTorch 관련 import
import torch # 기본 PyTorch 기능
from torch.utils.data import Dataset, DataLoader # 커스텀 데이터셋 및 배치 로딩
from sklearn.preprocessing import LabelEncoder # 라벨 → 숫자 변환
from sklearn.model_selection import train_test_split # 학습/검증 분리

# Huggingface Transformers 관련 import
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler  # 토크나이저, 분류 모델, 학습률 스케줄러

from torch.optim import AdamW  # 옵티마이저
from transformers import get_scheduler # 학습률 스케줄러

# 학습 진행률 시각화 관련 import
from tqdm import tqdm # 진행률 표시

# 시스템 관련 import
import os # 파일 시스템 관련 처리

# ------------------------------ #
# 설정
# ------------------------------ #
model_name = "madatnlp/km-bert" # 의학 특화 BERT
csv_path = "/content/drive/MyDrive/.ipynb_model_data/generated_disease_sentences_v2.csv"
batch_size = 16 # 배치 크기
num_epochs = 3 # 학습 횟수
max_len = 512 # 입력 문장의 최대 길이
lr = 2e-5 # 학습률 (learning rate)

# ✅ Google Drive 경로 설정
save_dir = "/content/drive/MyDrive/model_checkpoints" # 저장 폴더
os.makedirs(save_dir, exist_ok=True) # 폴더 없으면 생성

# ------------------------------ #
# 1. 데이터 로딩 및 전처리
# ------------------------------ #
df = pd.read_csv(csv_path) # CSV 파일에서 데이터프레임 로드
num_records = len(df) # 레코드 수 계산 (헤더 제외)

label_encoder = LabelEncoder() # 질병명 라벨 인코더 생성
df["label"] = label_encoder.fit_transform(df["disease_name"]) # 질병명을 숫자 라벨로 변환
num_classes = len(label_encoder.classes_) # 전체 클래스(질병) 개수 계산

# 학습/검증 데이터 분리 (80% 학습, 20% 검증)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["generated_sentence"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# ------------------------------ #
# 2. 토크나이저 및 데이터셋 정의
# ------------------------------ #
# KM-BERT 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained(model_name)

# 커스텀 데이터셋
class DiseaseDataset(Dataset):
    def __init__(
        self,
        texts, # 입력 문장 리스트
        labels # 정답 라벨 리스트
    ):
        # 토큰화 및 패딩
        self.encodings = tokenizer(
            texts,
            truncation=True, # max_length보다 길면 자름
            padding=True, # 짧은 문장은 max_length에 맞게 패딩
            max_length=max_len # 최대 길이를 지정
        )
        self.labels = labels # 정답 라벨 저장

    # 데이터셋에서 특정 인덱스에 해당하는 데이터를 반환
    def __getitem__(
        self,
        idx # 요청한 데이터의 인덱스
    ):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()} # 입력값을 텐서로 변환
        item["labels"] = torch.tensor(self.labels[idx]) # 라벨 추가
        return item # 하나의 학습 샘플 반환

    def __len__(self):
        return len(self.labels)

# 데이터셋 생성
train_dataset = DiseaseDataset(train_texts, train_labels) # 학습용 데이터셋
val_dataset = DiseaseDataset(val_texts, val_labels) # 검증용 데이터셋

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 학습용 데이터로더
val_loader = DataLoader(val_dataset, batch_size=batch_size) # 검증용 데이터로더

# ------------------------------ #
# 3. 모델 구성 및 최적화기
# ------------------------------ #
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes) # KM-BERT 분류 모델 초기화 (클래스 수 지정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 여부 설정
model.to(device) # 모델을 GPU 또는 CPU에 로드

optimizer = AdamW(model.parameters(), lr=lr) # Adamw 옵티마이저 설정
lr_scheduler = get_scheduler( # 학습률 스케줄러 설정
    "linear", # 선형 감소 스케줄
    optimizer=optimizer, # 적용할 옵티마이저
    num_warmup_steps=0, # 워밍업 단계 없음
    num_training_steps=len(train_loader) * num_epochs # 전체 학습 스텝 수 설정
)

# ------------------------------ #
# 4. 체크포인트 로드 (이어 학습)
# ------------------------------ #
latest_epoch = 0 # 기본 시작 epoch를 0으로 설정
checkpoint_files = sorted([ # 체크포인트 파일들을 정렬하여 리스트로 저장
    f for f in os.listdir(save_dir) # 저장 디렉토리 내 파일 목록을 순회
    if f.startswith(f"disease_classifier_epoch") and f.endswith(f"_{num_records}.pt") # 지정된 형식의 파일만 필터링
])

if checkpoint_files: # 체크포인트 파일이 있는 경우
    latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1]) # 가장 마지막 체크포인트 경로
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False) # 체크포인트 로드
    model.load_state_dict(checkpoint["model_state_dict"]) # 모델 파라미터 로드
    label_encoder = checkpoint["label_encoder"] # 레이블 인코더 로드
    latest_epoch = checkpoint["epoch"] # 마지막 학습된 epoch 불러오기
    
    # 체크포인트 로드 성공 메시지 출력
    print(f"✅ 체크포인트 로드 완료: {latest_checkpoint} | 이어서 epoch {latest_epoch + 1}부터 시작")
else: # 체크포인트 파일이 없는 경우 초기 학습 시작 안내
    print("ℹ️ 기존 체크포인트 없음. 처음부터 학습 시작")

# ------------------------------ #
# 5. 학습 루프
# ------------------------------ #
for epoch in range( # 학습을 재개할 epoch부터 num_epochs까지 반복
    latest_epoch, 
    num_epochs
):
    model.train() # 모델을 학습 모드로 설정
    total_loss = 0  # 에폭 전체 손실 누적
    correct = 0  # 정답 맞춘 개수
    total = 0  # 전체 샘플 개수

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"): # 학습 데이터 반복
        batch = {k: v.to(device) for k, v in batch.items()} # 배치를 GPU 또는 CPU로 이동
        outputs = model(**batch) # 모델에 배치 입력
        loss = outputs.loss # 손실 계산

        loss.backward() # 역전파 수행
        optimizer.step() # 가중치 업데이트
        lr_scheduler.step() # 학습률 스케줄러 업데이트
        optimizer.zero_grad() # 기울기 초기화

        total_loss += loss.item() # 손실 누적

        preds = torch.argmax(outputs.logits, dim=1) # 각 샘플에 대해 예측한 클래스 인덱스 추출
        labels = batch["labels"] # 정답 레이블 추출

        correct += (preds == labels).sum().item() # 맞춘 샘플 수 누적
        total += labels.size(0) # 전체 샘플 수 누적

    accuracy = correct / total # 정확도 계산
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}") # 현재 에폭의 손실과 정확도 출력

    # 에폭별 체크포인트 파일 경로 생성
    checkpoint_path = os.path.join(save_dir, f"disease_classifier_epoch{epoch+1}_{num_records}.pt") 
    torch.save({ # 모델 체크포인트 저장 
        "epoch": epoch + 1, # 현재 에폭 번호
        "model_state_dict": model.state_dict(), # 모델 가중치 저장
        "label_encoder": label_encoder, # 라벨 인코더 저장
    }, checkpoint_path)
    
    # 저장 완료 로그 출력
    print(f"🔄 중간 저장 완료: {checkpoint_path}") 

# ------------------------------ #
# 6. 최종 모델 저장
# ------------------------------ #
# 최종 모델 저장 경로 설정
final_model_path = os.path.join(save_dir, f"disease_classifier_final_{num_records}.pt")

torch.save({ # 최종 모델 저장
    "model_state_dict": model.state_dict(), # 모델의 학습된 가중치 저장
    "label_encoder": label_encoder, # 라벨 인코더 객체 저장
}, final_model_path) 

# 저장 완료 메시지 출력
print(f"✅ 모델 저장 완료: {final_model_path}")
