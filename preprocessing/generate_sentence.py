import random

class SentenceGenerator:
    def __init__(
        self, 
        disease_templates, # 질병 관련 문장 템플릿 (말투별로 구분된 딕셔너리)
        symptom_templates # 증상 관련 문장 템플릿 (말투별로 구분된 딕셔너리)
    ):
        self.disease_templates = disease_templates # 질병 템플릿 저장
        self.symptom_templates = symptom_templates # 증상 템플릿 저장
        # 생성된 문장들을 저장하는 집합 (중복 방지) 
        self.generated_sentences = set() 
    
    def filter_symptoms(
        self, 
        symptom_names # 증상 이름들의 리스트 (ex: ["기침", "두통", "", "발열"])
    ):
        # 빈 문자열("")을 제거하고, 유효한 증상 이름만 리스트로 반환
        return [s for s in symptom_names if s]  
    
    def generate_sentence(
        self, 
        disease_name, # 질병명 (ex: "독감", "감기")
        selected_symptoms, # 선택된 증상 목록 ( ex: ["기침", "발열"])
        style_type, # 문장 스타일 타입 ("disease" 또는 "symptom" 중 하나)
        tone_type, # 말투 스타일 타입 ("formal", "informal", "awkward", "lazy" 중 하나)
        split_symptoms, # 증상 목록을 나누어서 사용할지 여부 (true/false)
        min_symptoms, # 한 문장에 포함할 최소 증상 수
        max_symptoms # 한 문장에 포함할 최대 증상 수
    ):
        # 선택된 증상들의 개수 계산
        symptom_count = len(selected_symptoms)
        
        if symptom_count > max_symptoms: # 선택된 증상 수가 최대 증상 수를 초과하는 경우
            selected_symptoms = selected_symptoms[:max_symptoms] # 최대 개수만큼만 자름
        if symptom_count < min_symptoms: # 선택된 증상 수가 최소 증상 수보다 작은 경우
            selected_symptoms = selected_symptoms[:min_symptoms] # 최소 개수만큼만 사용

        # 스타일 타입에 따라 사용할 템플릿 풀 선택
        if style_type == "disease":
            # 질병 문장을 생성할 경우, 해당 말투(tone_type)에 맞는 질병 템플릿 사용
            template_pool = self.disease_templates[tone_type]
        else:
            # 증상 문장을 생성할 경우, 해당 말투(tone_type)에 맞는 증상 템플릿 사용
            template_pool = self.symptom_templates[tone_type]
        
        # 질병 문장을 생성할 경우
        if style_type == "disease": 
            # 템플릿 풀에서 랜덤으로 하나 선택 (ex. "{disease_name}이(가) 의심됩니다.")
            template = random.choice(template_pool)
            
            # 증상 리스트가 비어 있지 않으면 랜덤으로 하나 선택, 없으면 빈 문자열
            symptom_text = random.choice(selected_symptoms) if selected_symptoms else ""
            
            # 선택한 템플릿에 disease_name과 symptom_text를 채워 문장 생성
            sentence = template.format(symptom_text=symptom_text, disease_name=disease_name)
            
        # 증상 문장을 생성할 경우
        else:
            if not selected_symptoms: # 증상 목록이 비어있는 경우 
                # 해당 말투(tone_type)에 맞는 증상 템플릿 중에서 랜덤으로 하나 선택
                template = random.choice(self.symptom_templates[tone_type])
                # 선택한 템플릿에 disease_name만 채워서 문장 완성
                sentence = template.format(disease_name=disease_name)
                
            else: # 증상 목록이 비어있지 않은 경우
                # split_symptoms가 True이고, 선택된 증상이 2개 이상일 경우
                if split_symptoms and len(selected_symptoms) >= 2:
                    # 각 증상에 대해 말투(tone_type)에 맞는 템플릿을 랜덤 선택하여 문장 생성
                    symptom_sentences = [
                        random.choice(self.symptom_templates[tone_type]).format(symptom_text=symptom)
                        for symptom in selected_symptoms
                    ]
                        
                    random.shuffle(symptom_sentences) # 생성된 증상 문장들을 랜덤하게 섞음
                    
                    # 섞은 문장들을 공백(" ")으로 이어붙여 하나의 문장으로 만듦
                    sentence = " ".join(symptom_sentences)
                    
                # split_symptoms가 False거나, 선택된 증상이 1개일 경우
                else:
                    # 선택된 증상들을 쉼표(", ")로 구분하여 하나의 문자열로 합침
                    symptom_text = ", ".join(selected_symptoms)
                    
                    # 템플릿 풀(template_pool)에서 랜덤으로 하나 선택
                    template = random.choice(template_pool)
                    
                    # 선택한 템플릿에 symptom_text와 disease_name을 채워 문장 완성
                    sentence = template.format(symptom_text=symptom_text, disease_name=disease_name)

        # 이미 생성된 문장인 경우
        if sentence in self.generated_sentences:
            return None # 중복 방지를 위해 None 반환
        
        # 생성한 문장을 기록 (중복 방지를 위해 저장)
        self.generated_sentences.add(sentence)

        # 최종 생성된 문장을 반환
        return sentence

    def generate_sentences(
        self, 
        diseases, # 질병 목록 (ex: [{"name": "감기", "symptoms": [1, 2, 3], ...}])
        symptom_id_to_name, # 증상 ID를 증상 이름으로 매핑한 딕셔너리 (ex: {1: "기침", 2: "발열"})
        sentences_per_disease=100 # 질병 하나당 생성할 문장 수 (기본값 50개)
    ):
        results = [] # 생성된 문장을 저장할 리스트 초기화

        for disease in diseases: # 질병 목록을 하나씩 순회
            # 질병 이름 가져오기
            disease_name = disease['name'] 
            
            # 해당 질병에 연결된 증상 ID 목록 가져오기
            symptom_ids = disease['symptoms']
            
            # 증상 ID를 증상 이름으로 변환
            symptom_names = [symptom_id_to_name.get(id) for id in symptom_ids]
            
            # 변환된 증상 이름 리스트에서 빈 값 제거
            symptom_names = self.filter_symptoms(symptom_names)

            # 증상 이름 리스트가 비어 있는 경우
            if not symptom_names:
                print(f"Warning: Disease '{disease_name}' has no symptoms. Skipping.") # 경고 출력
                continue # 해당 질병은 건너뜀

            # 증상 개수 계산
            symptom_count = len(symptom_names) 
            
            # 한 문장에서 사용할 최소 증상 수 설정 (항상 1개 이상)
            min_symptoms = 1 
            
            # 한 문장에서 사용할 최대 증상 수 설정 (최대 3개 또는 증상 수 중 작은 값)
            max_symptoms = min(symptom_count, 3)  

            # 사용할 증상 개수 선택지를 리스트로 생성 (ex: [1, 2, 3])
            symptom_count_choices = list(range(min_symptoms, max_symptoms + 1))  

            # 만약 선택할 증상 개수가 하나도 없는 경우
            if not symptom_count_choices:
                print(f"Warning: Disease '{disease_name}' does not have enough symptoms to meet the range. Skipping.") # 경고 출력
                continue # 해당 질병은 건너뜀

            # 질병당 생성할 문장 수만큼 반복
            for _ in range(sentences_per_disease): 
                
                # 생성할 문장의 스타일 타입 선택 (질병 문장 또는 증상 문장)
                style_choices = ["disease", "symptom"]
                style_type = random.choice(style_choices)

                # 생성할 문장의 말투(톤) 선택 (존댓말형, 반말형, 어색형, 단답형)
                tone_choices = ["formal", "informal", "awkward", "lazy"]
                tone_type = random.choice(tone_choices)

                # 증상 나누기 여부 초기화
                split_symptoms = False # 기본값 False
                if style_type == "symptom": # 스타일이 증상(symptom)일 경우
                    split_choices = [True, False] 
                    split_symptoms = random.choice(split_choices) # 증상 나누기를 랜덤으로 선택

                # 사용할 증상 개수를 symptom_count_choices 중 랜덤하게 선택
                symptom_count = random.choice(symptom_count_choices)  
                
                # 선택한 증상 개수만큼 증상 이름들 중 랜덤하게 뽑기
                selected_symptoms = random.sample(symptom_names, symptom_count)  

                # 설정한 조건(질병명, 증상 목록, 스타일, 말투 등)에 맞춰 문장 생성
                sentence = self.generate_sentence(disease_name, selected_symptoms, style_type, tone_type, split_symptoms, min_symptoms, max_symptoms)
                
                # 문장이 정상적으로 생성되었으면 결과 리스트에 추가
                if sentence:
                    results.append({
                        "disease_id": disease['id'], # 질병 ID
                        "disease_name": disease_name, # 질병명
                        "generated_sentence": sentence # 생성된 문장
                    })

        return results

