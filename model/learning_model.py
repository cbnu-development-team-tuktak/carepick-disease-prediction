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
csv_path = "/content/drive/MyDrive/.ipynb_model_data/train_dataset.csv"
batch_size = 16 # 배치 크기
num_epochs = 15 # 학습 횟수
max_len = 512 # 입력 문장의 최대 길이
lr = 2e-5 # 학습률 (learning rate)

# Google Drive 경로 설정
save_dir = "/content/drive/MyDrive/model_checkpoints" # 저장 폴더
os.makedirs(save_dir, exist_ok=True) # 폴더 없으면 생성

# ------------------------------ #
# 1. 데이터 로딩 및 전처리
# ------------------------------ #
df = pd.read_csv(csv_path, header=None)  # 헤더 없는 CSV 읽기
df.columns = ["disease_name", "generated_sentence", "style_type"]  # 컬럼 수동 지정

num_records = len(df)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["disease_name"])
num_classes = len(label_encoder.classes_)

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

# KM-BERT 분류 모델 초기화 (클래스 수 지정)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 여부 설정
model.to(device) # 모델을 GPU 또는 CPU에 로드

optimizer = AdamW(model.parameters(), lr=lr) # Adamw 옵티마이저 설정
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

# ------------------------------ #
# 4. 체크포인트 로드 (이어 학습)
# ------------------------------ #
latest_epoch = 9
checkpoint_files = sorted([f for f in os.listdir(save_dir) if f.startswith(f"disease_classifier_epoch") and f.endswith(f"_{num_records}.pt")])

if checkpoint_files:
    latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1])
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    label_encoder = checkpoint["label_encoder"]
    latest_epoch = checkpoint["epoch"]
    print(f"✅ 체크포인트 로드 완료: {latest_checkpoint} | 이어서 epoch {latest_epoch + 1}부터 시작")
else:
    print("ℹ️ 기존 체크포인트 없음. 처음부터 학습 시작")

# ------------------------------ #
# 5. 학습 루프
# ------------------------------ #
for epoch in range(latest_epoch, num_epochs):
    model.train()
    total_loss = 0  # 에폭 전체 손실 누적
    correct = 0  # 정답 맞춘 개수
    total = 0  # 전체 샘플 개수

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # 예측값 계산 (softmax 없이 argmax만으로 가능)
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch["labels"]

        correct += (preds == labels).sum().item()  # 정답 개수 누적
        total += labels.size(0)  # 전체 개수 누적

    accuracy = correct / total  # 정확도 계산
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

    # 중간 저장
    checkpoint_path = os.path.join(save_dir, f"disease_classifier_epoch{epoch+1}_{num_records}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "label_encoder": label_encoder,
    }, checkpoint_path)
    print(f"🔄 중간 저장 완료: {checkpoint_path}")

# ------------------------------ #
# 6. 최종 모델 저장
# ------------------------------ #
final_model_path = os.path.join(save_dir, f"disease_classifier_final_{num_records}.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "label_encoder": label_encoder,
}, final_model_path)

print(f"✅ 모델 저장 완료: {final_model_path}")
