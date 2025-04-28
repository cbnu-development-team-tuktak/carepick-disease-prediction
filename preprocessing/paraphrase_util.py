# paraphrase_util.py

# Requests 및 데이터 핸들링 관련 import
import torch # PyTorch 텐서 및 디바이스(CPU/GPU) 제어 라이브러리
from transformers import AutoTokenizer, AutoModel # Huggingface에서 사전 학습된 BERT 모델과 토크나이저 로드
from sklearn.metrics.pairwise import cosine_similarity # 벡터 간 유사도 계산을 위한 라이브러리

# ------------------------- #
# 1. BERT 모델 로딩
# ------------------------- #
model_name = "madatnlp/km-bert" # 사용할 KM-BERT 모델 이름 설정
tokenizer = AutoTokenizer.from_pretrained(model_name) # KM-BERT 토크나이저 불러오기
bert_model = AutoModel.from_pretrained(model_name) # KM-BERT 모델 불러오기
bert_model.eval() # 모델을 평가 모드로 전환 (학습 X, 추론 전용)

# CUDA (GPU) 사용 가능 여부에 따라 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device) # 모델을 설정한 디바이스로 이동

# ------------------------- #
# 2. 문장을 BERT 임베딩으로 변환하는 함수
# ------------------------- #
def get_embedding(
    text # 임베딩할 텍스트 (문자열)
):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device) # 입력 텍스트 토크나이즈
    with torch.no_grad():
        outputs = bert_model(**inputs) # BERT 모델 추론
        embedding = outputs.last_hidden_state[:, 0, :] # [CLS] 토큰 벡터 사용
    return embedding.cpu() # 결과를 CPU로 이동하여 반환

# ------------------------- #
# 3. 주어진 후보 리스트 중 가장 유사한 표현 찾기
# ------------------------- #
def find_most_similar(
    word, # 기준 단어 (문자열)
    candidates # 후보 단어 리스트 (ex: ["비루", "코막힘", "콧물이 흐름"])
):
    word_emb = get_embedding(word) # 기준 단어 임베딩
    candidate_embs = torch.cat([get_embedding(cand) for cand in candidates], dim=0) # 후보 단어들 임베딩
    similarities = cosine_similarity(word_emb.numpy(), candidate_embs.numpy()) # 코사인 유사도 계산
    best_idx = similarities.argmax() # 가장 유사한 후보 인덱스
    return candidates[best_idx] # 가장 유사한 후보 단어 반환

# ------------------------- #
# 4. 증상 리스트에서 각 증상별 유사 증상 추천 함수
# ------------------------- #
def build_paraphrase_candidates(
    symptom_list, # 증상명 리스트 (ex: ["콧물", "두통", "기침", ...])
    similarity_threshold=0.8 # 유사도 임계값 (default 0.8)
):
    embeddings = torch.cat([get_embedding(s) for s in symptom_list], dim=0) # 증상 전체 임베딩
    similarity_matrix = cosine_similarity(embeddings.numpy()) # 코사인 유사도 계산
    candidates_dict = {} # 결과 저장할 딕셔너리

    for idx, symptom in enumerate(symptom_list):
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # 유사도 높은 순 정렬

        top_similar = []
        for i, score in sim_scores[1:]: # 자기 자신 제외 (0번째는 자기 자신)
            if score >= similarity_threshold:
                top_similar.append(symptom_list[i])

        candidates_dict[symptom] = top_similar if top_similar else [symptom] # 유사 증상이 없으면 자기 자신

    return candidates_dict # { "콧물": ["코막힘", "비루"], "두통": ["편두통"], ... }
