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
checkpoint_path = "/content/drive/MyDrive/model_checkpoints/disease_classifier_final_23143.pt" # 학습된 모델의 체크포인트 경로
max_len = 512 # 입력 시퀀스의 최대 길이
batch_size = 16 # 배치 시퀀스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 사용할 디바이스 (GPU 사용 가능 시 CUDA)

# ---------------------------- #
# 토크나이저 및 데이터셋 정의
# ---------------------------- #
tokenizer = BertTokenizer.from_pretrained(model_name) # 사전 학습된 BERT 모델 토크나이저 로드

class TestDataset(
    Dataset # Python Dataset 상속
):
    def __init__(
        self,
        texts, # 입력 텍스트 리스트
        data # label, url 등 평가용 원본 데이터 리스트
    ):
        self.data = data # 평가 시 실제 라벨과 URL 추출을 위해 원본 데이터를 보존
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
# Top-k 정확도 계산
# ---------------------------- #
def compute_topk_accuracy(
    true_labels, # 실제 정답 질병명 리스트
    topk_preds, # 모델이 예측한 Top-k 질병 리스트 (예: [["감기", "폐렴", "비염"], ...])
    k=3 # Top-k 값
):
    correct = 0 # Top-k 안에 정답이 포함된 개수
    
    # 각 정답과 예측 리스트 순회
    for true_label, pred_list in zip(true_labels, topk_preds):
        if true_label in pred_list[:k]: # 정답이 Top-k 예측 안에 있는 경우
            correct += 1 # correct 값 증가
            
    # Top-k 정확도 계산
    return correct / len(true_labels)

# ---------------------------- #
# MRR 정확도 계산
# ---------------------------- #
def compute_mrr(
    true_labels, # 실제 정답 질병명 리스트
    topn_preds # 모델이 예측한 Top-k 질병 리스트 (예: [["감기", "폐렴", "비염"], ...])
):
    mrr_total = 0 # 전체 reciprocal rank의 합 초기화
    
    # 각 정답과 예측 리스트 순회 
    for true_label, pred_list in zip(true_labels, topn_preds): 
        # 정답이 예측 리스트 내 몇 번째(index + 1)에 위치하는지 찾음 (없으면 0)
        rank = next((i + 1 for i, pred in enumerate(pred_list) if pred == true_label), 0)
        
        # reciprocal rank 계산 (정답이 없으면 0)
        mrr_total += 1 / rank if rank else 0
        
    return mrr_total / len(true_labels) # 평균 reciprocal rank 반환

# ---------------------------- #
# 결과 판단
# ---------------------------- #
def classify_result(
    topk_acc,  # Top-k 정확도 (0 또는 1)
    mrr_score  # Mean Reciprocal Rank (0.0 ~ 1.0)
):
    # 정답이 Top-k 안에 있고, 그중에서 높은 순위 (1~2등)에 위치한 경우
    if topk_acc == 1 and mrr_score >= 0.5:
        return "정확"
    
    # 정답이 Top-k 안에 있으나, 낮은 순위 (3등 이하)에 위치한 경우
    elif topk_acc == 1 and mrr_score < 0.5:
        return "검증 필요"
    
    # 정답이 Top-k 예측 결과에 아예 포함되지 않은 경우
    else:
        return "부정확"
    
# ---------------------------- #
# Top-K 예측 및 평가
# ---------------------------- #
def predict_and_evaluate(
    model, # 학습된 모델
    dataloader, # 배치 단위 입력용 데이터로더
    label_encoder, # 라벨 인코더 (인덱스 → 질병명 디코딩용)
    device, # 실행 디바이스 ('cuda' 또는 'cpu')
    k = 3 # Top-k 설정값 (기본값: 3)
):
    model.eval()
    results = [] # 최종 Top-N 질병 예측 결과를 저장할 리스트
    with torch.no_grad(): # 그래디언트 계산 비활성화 (추론 모드)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="예측 중")):
            batch = {k: v.to(device) for k, v in batch.items()} # 입력 데이터를 GPU 또는 CPU로 이동
            outputs = model(**batch) # 모델에 입력하여 출력(로짓) 획득
            probs = softmax(outputs.logits, dim=1) # 로짓에 소프트맥스를 적용하여 확률로 변환

            topk = torch.topk(probs, k, dim=1) # 각 샘플에서 확률이 높은 상위 N개 인덱스 추출
            top_indices = topk.indices.cpu().tolist() # 예측 인덱스를 리스트로 변환
            top_values = topk.values.cpu().tolist() # 예측 확률을 리스트로 변환

            for i, (indices, scores) in enumerate(zip(top_indices, top_values)):
                # 예측된 질병 인덱스를 실제 질병명으로 인코딩 (예: [42, 13, 7] → ['감기', '폐렴', '비염'])
                diseases = label_encoder.inverse_transform(indices)
                
                # 예측된 질병명과 각 확률을 "질병명 (0.xx)" 형식으로 문자열 결합
                predicted_str = ", ".join(f"{d} ({s:.2f})" for d, s in zip(diseases, scores))

                # 전체 데이터셋 상의 원래 인덱스를 계산 (현재 배치 기준 offset 고려)
                idx = batch_idx * dataloader.batch_size + i
                
                # 실제 정답 질병 라벨 추출
                true_label = dataloader.dataset.data[idx]["label"]
                
                # 입력 질문 텍스트 추출
                input_text = dataloader.dataset.data[idx]["text"]
                
                # URL이 포함되어 있다면 함께 추출, 없으면 빈 문자열로 대체
                url = dataloader.dataset.data[idx].get("url", "")  

                # top-k accuracy 계산 (정답이 Top-k 예측 안에 있는지 여부: 0 또는 1)
                topk_acc = compute_topk_accuracy([true_label], [diseases], k)
                
                # MRR 계산 (정답의 순위에 따른 reciprocal rank)
                mrr = compute_mrr([true_label], [diseases])
                
                # 정답이 몇 번째로 예측됐는지 순위 계산
                try: # 정답 질병명이 예측된 질병 리스트 안에 있는 경우
                    rank = diseases.tolist().index(true_label) + 1 # 몇 번째에 위치해있는지 계산
                except ValueError: # 정답 질병명이 예측된 Top-k 리스트에 없는 경우
                    rank = 0 # 순위를 0으로 설정

                # 평가 결과 분류 ("정확", "검증 필요", "부정확")
                result = classify_result(topk_acc, mrr)

                # 결과 누적
                results.append({
                    "label": true_label, # 실제 정답 질병명
                    "text": input_text, # 입력된 증상 텍스트
                    "url": url, # 원본 질문 URL
                    "topk_predicted_diseases": predicted_str, # 예측된 top-k 질병 목록과 확률 
                    "topk_accuracy": topk_acc, # 정답이 top-k 안에 포함되었는지 여부 (0 또는 1)
                    "mrr_score": round(mrr, 4), # MRR 점수 (소수점 4자리로 반환)
                    "rank": rank, # 정답 질병의 예측 결과 내 위치
                    "result": result # 종합 평가 결과 ("정확", "검증 필요", "부정확")
                })

    # 누적된 예측 및 평가 결과 리스트를 DataFrame으로 반환
    return pd.DataFrame(results)



# ---------------------------- #
# 질병 리스트 불러오기
# ---------------------------- #
with open("csv/disease_list.csv", encoding="utf-8-sig") as f:
    disease_list = [line.strip() for line in f if line.strip()]
    
# ---------------------------- #
# 결과 저장
# ---------------------------- #
def save_all_predictions(
    model, # 학습된 모델 
    label_encoder, # 라벨 인코더 (인덱스 → 질병명 디코딩용)
    device, # 실행 디바이스 ('cuda' 또는 'cpu')
    disease_list, # 테스트 대상 질병명 리스트
    k = 3 # Top-k 설정값 (기본값: 3)
):
    all_results = [] # 모든 예측 결과 리스트
    
    for disease in disease_list: # 질병 리스트를 순회하며 각 질병별 테스트 데이터에 대한 예측 수행
        disease = disease.strip('"') # 큰따옴표는 전부 제거
    
        # 질병명 기준으로 테스트용 CSV 파일 경로 구성
        test_csv_path = f"/content/drive/MyDrive/.ipynb_model_data/test_dataset_{disease}.csv"
        
        # 해당 테스트 파일이 존재하지 않는 경우
        if not os.path.exists(test_csv_path):
            print(f"[SKIP] {disease}: 테스트 파일 없음") # 경고 로그 출력
            continue # 해당 질병 테스트는 건너뛰기

        # 예측 진행 로그 출력
        print(f"[INFO] {disease} 예측 중...")
        
        # CSV 로드 및 텍스트 추출
        df = pd.read_csv(test_csv_path) # 해당 질병의 테스트 CSV 파일 불러오기
        test_texts = df["text"].tolist() # 텍스트 컬럼만 리스트로 추출

        # 데이터셋 및 데이터로더 구성
        test_dataset = TestDataset(test_texts, df.to_dict("records")) # 레이블 포함된 테스트 데이터셋 구성
        test_loader = DataLoader(test_dataset, batch_size=batch_size) # 배치 단위로 로딩할 DataLoader 구성

        # 예측 및 평가 실행
        prediction_df = predict_and_evaluate(model, test_loader, label_encoder, device, k) # 모델을 이용해 예측 및 평가 수행
        all_results.append(prediction_df) # 결과를 리스트에 누적

    # 모든 결과 병합 후 저장
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 저장할 파일 경로 설정 (Top-k 값을 파일명에 반영)
    output_path = f"/content/drive/MyDrive/predictions_disease_top{k}.csv"
    
    # 최종 예측 결과를 CSV 파일로 저장 
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 저장 완료 메시지 출력
    print(f"\n✅ 전체 예측 완료 → {output_path}")