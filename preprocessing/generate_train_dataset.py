# Requests 및 데이터 핸들링 관련 import
import requests  # HTTP 요청을 보내기 위한 라이브러리
import pandas as pd  # 데이터프레임(DataFrame) 처리 라이브러리

from generate_sentence import SentenceGenerator  # 균형 잡힌 문장 생성 함수

# 질병 템플릿 관련 import
from template.disease_templates import (
    disease_templates_formal, # 존댓말형
    disease_templates_informal, # 반말형
    disease_templates_awkward, # 횡설수설형
    disease_templates_lazy # 단답형
)
# 증상 템플릿 관련 import
from template.symptom_templates import (
    symptom_templates_formal, # 존댓말형
    symptom_templates_informal, # 반말형
    symptom_templates_awkward, # 횡설수설형
    symptom_templates_lazy # 단답형
)

# ------------------------- #
# 1. 데이터 로딩
# ------------------------- #
def load_data(
    disease_url, # 질병 정보 API URL
    symptom_url # 증상 정보 API URL
):
    disease_response = requests.get(disease_url) # 질병 목록 API 요청
    symptom_response = requests.get(symptom_url) # 증상 목록 API 요청
    disease_response.raise_for_status() # 요청 실패 시 예외 발생
    symptom_response.raise_for_status() # 요청 실패 시 예외 발생

    diseases = disease_response.json()['content'] # 질병 목록 추출
    symptoms = symptom_response.json()['content'] # 증상 목록 추출

    # 질병 목록과 증상 목록을 반환
    return diseases, symptoms

# 증상 ID를 이름으로 매핑
def map_symptom_ids_to_names(
    symptoms # 증상 목록
):
    # {id: 이름} 형태의 딕셔너리 반환
    return {symptom['id']: symptom['name'] for symptom in symptoms}

# ------------------------- #
# 2. 문장 생성
# ------------------------- #
def generate_dataset(
    diseases, # 질병 목록
    symptom_id_to_name # { 증상 ID: 증상명 } 매핑 딕셔너리
):
    # 문장 생성기 초기화
    sentence_generator = SentenceGenerator(
        disease_templates={
            "formal": disease_templates_formal, # 존댓말형 
            "informal": disease_templates_informal, # 반말형
            "awkward": disease_templates_awkward, # 횡설수설형
            "lazy": disease_templates_lazy # 단답형
        },
        symptom_templates={
            "formal": symptom_templates_formal, # 존댓말형 
            "informal": symptom_templates_informal, # 반말형
            "awkward": symptom_templates_awkward, # 횡설수설형
            "lazy": symptom_templates_lazy # 단답형
        } 
    )

    # 생성된 문장을 저장할 리스트
    generated_data = []

    # 각 질병에 대해 반복
    for disease in diseases:
        # 질병에 연결된 증상 이름 추출
        symptom_names = [symptom_id_to_name.get(id) for id in disease['symptoms']]
        # 유효한 증상 이름만 필터링
        symptom_names = [name for name in symptom_names if name]

        if not symptom_names: # 증상이 없는 경우
            print(f"Warning: Disease '{disease['name']}' has no symptoms. Skipping.") # 경고 출력
            continue # 해당 질병은 건너뜀
        
        # 한 질병에 대해 여러 문장을 생성
        results = sentence_generator.generate_sentences(
            [disease], # 단일 질병 정보만 전달 
            symptom_id_to_name, # 증상 ID → 이름 매핑
        )

        # 생성된 문장 결과들을 반복
        for result in results:
            # 각 문장을 딕셔너리 형태로 저장
            generated_data.append({
                'disease_id': result['disease_id'], # 질병 ID
                'disease_name': result['disease_name'], # 질병명 
                'generated_sentence': result['generated_sentence'] # 생성된 문장
            })

    # 최종 결과를 데이터프레임 형태로 반환
    return pd.DataFrame(generated_data)

# ------------------------- #
# 3. CSV 저장
# ------------------------- #
def save_to_csv(
    df, # 저장할 데이터프레임
    filename # 저장할 파일 이름
):  
    # 인덱스 없이 UTF-8-SIG 인코딩으로 저장
    df.to_csv(filename, index=False, encoding='utf-8-sig')

# ------------------------- #
# 메인 실행 흐름
# ------------------------- #
if __name__ == "__main__":
    # 질병 정보 API 주소
    DISEASE_API_URL = "http://localhost:8080/api/diseases/processed?page=0&size=500"
    # 증상 정보 API 주소
    SYMPTOM_API_URL = "http://localhost:8080/api/symptoms?page=0&size=3000"

    # 질병 및 증상 데이터 로드
    diseases, symptoms = load_data(DISEASE_API_URL, SYMPTOM_API_URL)

    # 증상 ID → 증상명 매핑 생성
    symptom_id_to_name = map_symptom_ids_to_names(symptoms)

    # 문장 데이터셋 생성
    generated_df = generate_dataset(diseases, symptom_id_to_name)

    # 생성된 문장을 CSV 파일로 저장
    save_to_csv(generated_df, 'generated_disease_sentences.csv')
