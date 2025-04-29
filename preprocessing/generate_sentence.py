import random

class SentenceGenerator:
    def __init__(self, disease_templates, symptom_templates):
        self.disease_templates = disease_templates
        self.symptom_templates = symptom_templates
        self.generated_sentences = set()  # 생성된 문장들을 저장할 공간
    
    def filter_symptoms(self, symptom_names):
        return [s for s in symptom_names if s]  # 빈 문자열 제거
    
    def generate_sentence(self, disease_name, selected_symptoms, style_type, tone_type, split_symptoms, min_symptoms, max_symptoms):
        # 증상 개수 조정
        symptom_count = len(selected_symptoms)
        if symptom_count > max_symptoms:
            selected_symptoms = selected_symptoms[:max_symptoms]  # 최대 증상 개수로 제한
        if symptom_count < min_symptoms:
            selected_symptoms = selected_symptoms[:min_symptoms]  # 최소 증상 개수로 제한

        # 문장 생성 로직
        if style_type == "disease":
            template_pool = self.disease_templates[tone_type]
        else:
            template_pool = self.symptom_templates[tone_type]
        
        # 문장 생성
        if style_type == "disease":
            template = random.choice(template_pool)
            symptom_text = random.choice(selected_symptoms) if selected_symptoms else ""
            sentence = template.format(symptom_text=symptom_text, disease_name=disease_name)
        else:
            if not selected_symptoms:
                # 🔥 수정: 빈 증상이어도 tone_type에 맞게 템플릿 선택
                template = random.choice(self.symptom_templates[tone_type])
                sentence = template.format(disease_name=disease_name)
            else:
                if split_symptoms and len(selected_symptoms) >= 2:
                    # 🔥 수정: split 할 때도 tone_type 기반으로!
                    symptom_sentences = [
                        random.choice(self.symptom_templates[tone_type]).format(symptom_text=symptom)
                        for symptom in selected_symptoms
                    ]
                    random.shuffle(symptom_sentences)
                    sentence = " ".join(symptom_sentences)
                else:
                    symptom_text = ", ".join(selected_symptoms)
                    template = random.choice(template_pool)
                    sentence = template.format(symptom_text=symptom_text, disease_name=disease_name)

        # 중복된 문장이면 None 반환
        if sentence in self.generated_sentences:
            return None
        self.generated_sentences.add(sentence)

        return sentence

    def generate_sentences(self, diseases, symptom_id_to_name, sentences_per_disease=20):
        results = []

        # 각 질병에 대해 문장 생성
        for disease in diseases:
            disease_name = disease['name']
            symptom_ids = disease['symptoms']
            symptom_names = [symptom_id_to_name.get(id) for id in symptom_ids]
            symptom_names = self.filter_symptoms(symptom_names)

            # 증상 목록이 비어있을 경우 예외 처리
            if not symptom_names:
                print(f"Warning: Disease '{disease_name}' has no symptoms. Skipping.")
                continue

            # 증상 개수 설정: disease['symptoms'] 길이를 기반으로 설정
            symptom_count = len(symptom_names)  # 질병에 연결된 증상의 실제 개수 사용
            min_symptoms = 1  # 최소 증상 개수 설정
            max_symptoms = symptom_count  # 최대 증상 개수는 해당 질병에 연결된 증상 개수로 설정

            # 증상 개수를 고려하여 문장 생성
            symptom_count_choices = list(range(min_symptoms, max_symptoms + 1))  # 증상 개수의 범위 설정

            # 가능한 증상 개수가 없으면 건너뛰기
            if not symptom_count_choices:
                print(f"Warning: Disease '{disease_name}' does not have enough symptoms to meet the range. Skipping.")
                continue

            # 각 문장마다 스타일, 말투, split 여부를 랜덤하게 설정
            for _ in range(sentences_per_disease):  # 균등하게 문장 20개 생성
                # 스타일 선택
                style_choices = ["disease", "symptom"]
                style_type = random.choice(style_choices)

                # 말투 선택
                tone_choices = ["formal", "informal", "awkward", "lazy"]
                tone_type = random.choice(tone_choices)

                # split 여부 선택 (symptom 스타일일 때만)
                split_symptoms = False
                if style_type == "symptom":
                    split_choices = [True, False]
                    split_symptoms = random.choice(split_choices)

                # 증상 개수 랜덤 선택
                symptom_count = random.choice(symptom_count_choices)  # 증상 개수 랜덤 선택
                selected_symptoms = random.sample(symptom_names, symptom_count)  # 선택된 증상 개수만큼 문장 생성

                # 문장 생성
                sentence = self.generate_sentence(disease_name, selected_symptoms, style_type, tone_type, split_symptoms, min_symptoms, max_symptoms)
                
                if sentence:
                    results.append({
                        "disease_id": disease['id'],
                        "disease_name": disease_name,
                        "generated_sentence": sentence
                    })

        return results

