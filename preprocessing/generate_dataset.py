# ------------------------- #
# Requests 및 데이터 핸들링 관련 import
# ------------------------- #
import requests  # HTTP 요청을 보내기 위한 라이브러리
import pandas as pd  # 데이터프레임(DataFrame) 처리 라이브러리

from generate_sentence import SentenceGenerator  # 균형 잡힌 문장 생성 함수

# 질병 템플릿 관련 import
from template.disease_templates import (
    disease_templates_formal,
    disease_templates_informal,
    disease_templates_awkward,
    disease_templates_lazy
)
from template.symptom_templates import (
    symptom_templates_formal,
    symptom_templates_informal,
    symptom_templates_awkward,
    symptom_templates_lazy
)

# ------------------------- #
# 1. 데이터 로딩 함수
# ------------------------- #
def load_data(disease_url, symptom_url):
    """주어진 URL에서 질병 및 증상 데이터를 로드합니다."""
    disease_response = requests.get(disease_url)
    symptom_response = requests.get(symptom_url)
    disease_response.raise_for_status()
    symptom_response.raise_for_status()

    diseases = disease_response.json()['content']
    symptoms = symptom_response.json()['content']

    return diseases, symptoms

def map_symptom_ids_to_names(symptoms):
    """증상 리스트를 ID에서 이름으로 매핑합니다."""
    return {symptom['id']: symptom['name'] for symptom in symptoms}

# ------------------------- #
# 2. 문장 생성 함수
# ------------------------- #
def generate_dataset(diseases, symptom_id_to_name):
    """질병 데이터를 바탕으로 문장을 생성하고, DataFrame으로 반환합니다."""
    sentence_generator = SentenceGenerator(
        disease_templates={
            "formal": disease_templates_formal,
            "informal": disease_templates_informal,
            "awkward": disease_templates_awkward,
            "lazy": disease_templates_lazy
        },
        symptom_templates={
            "formal": symptom_templates_formal,
            "informal": symptom_templates_informal,
            "awkward": symptom_templates_awkward,
            "lazy": symptom_templates_lazy
        }
    )

    generated_data = []

    for disease in diseases:
        # 증상 개수 범위 설정: disease['symptoms']의 길이에 따라 동적으로 설정
        symptom_names = [symptom_id_to_name.get(id) for id in disease['symptoms']]
        symptom_names = [name for name in symptom_names if name]

        if not symptom_names:  # 증상 목록이 비어있을 경우 예외 처리
            print(f"Warning: Disease '{disease['name']}' has no symptoms. Skipping.")
            continue
        
        results = sentence_generator.generate_sentences(
            [disease],
            symptom_id_to_name,
        )

        # 문장 생성 결과 추가
        for result in results:
            generated_data.append({
                'disease_id': result['disease_id'],
                'disease_name': result['disease_name'],
                'generated_sentence': result['generated_sentence']
            })

    # 결과를 Pandas DataFrame으로 반환
    return pd.DataFrame(generated_data)

# ------------------------- #
# 3. CSV 저장 함수
# ------------------------- #
def save_to_csv(df, filename):
    """DataFrame을 CSV 파일로 저장합니다."""
    df.to_csv(filename, index=False, encoding='utf-8-sig')

# ------------------------- #
# 메인 실행 흐름
# ------------------------- #
if __name__ == "__main__":
    DISEASE_API_URL = "http://localhost:8080/api/diseases/processed?page=0&size=500"
    SYMPTOM_API_URL = "http://localhost:8080/api/symptoms?page=0&size=3000"

    # 질병 및 증상 데이터 로딩
    diseases, symptoms = load_data(DISEASE_API_URL, SYMPTOM_API_URL)

    # 증상 ID → 이름 매핑 생성
    symptom_id_to_name = map_symptom_ids_to_names(symptoms)

    # 질병-증상 기반 문장 데이터셋 생성
    generated_df = generate_dataset(diseases, symptom_id_to_name)

    # 생성된 데이터셋을 CSV 파일로 저장
    save_to_csv(generated_df, 'generated_disease_sentences_v1.csv')
