import os
import pandas as pd

# 카테고리별 질병 목록 사전
category_disease_map = {
    "감염성 질환 및 바이러스성 질환": [
        "B형 간염", "C형 간염", "감기", "결핵", "대상포진", "방광염", "부비동염(축농증)",
        "세균성 장염", "수두", "수족구병", "신우신염", "요로감염", "위장염", "인플루엔자",
        "장티푸스", "코로나19", "편도염", "폐렴", "후두염"
    ],
    "호흡기계 질환": [
        "기관지염", "만성 폐쇄성 폐질환(COPD)", "비후성 비염", "알레르기 비염", "천식"
    ],
    "소화기계 질환": [
        "간경변", "과민성대장증후군(IBS)", "궤양성 대장염", "담석증", "변비",
        "소화불량", "십이지장궤양", "역류성 식도염(GERD)", "위궤양", "위염",
        "장염", "지방간", "치질(치핵)", "크론병"
    ],
    "근골격계 질환": [
        "경추통(목디스크)", "골다공증", "골절", "관절염(골관절염)", "근막통증증후군",
        "류마티스 관절염", "무릎관절염", "섬유근육통", "손목터널증후군", "오십견",
        "요통", "족저근막염", "척추관 협착증", "척추측만증", "테니스엘보",
        "통풍", "허리디스크(요추간판 탈출증)", "회전근개파열"
    ],
    "신경계/정신건강 질환": [
        "ADHD", "PTSD", "간질(뇌전증)", "강박장애", "공황장애",
        "뇌경색", "뇌졸중(중풍)", "뇌출혈", "두통", "불면증",
        "불안장애", "수면무호흡증", "어지럼증(현훈)", "우울증", "조현병",
        "치매", "파킨슨병", "편두통"
    ],
    "내분비/대사질환": [
        "갑상선 기능저하증", "갑상선 기능항진증", "고혈압", "당뇨병", "대사증후군",
        "비만", "이상지질혈증(고지혈증)", "인슐린저항성", "저혈당증", "쿠싱증후군"
    ],
    "피부질환": [
        "건선", "두드러기", "무좀", "백반증", "사마귀",
        "아토피 피부염", "여드름", "접촉성 피부염", "지루성 피부염",
        "탈모증", "티눈", "피부양진"
    ],
    "안과 질환": [
        "결막염", "녹내장", "눈다래끼(맥립종)", "백내장", "시신경염",
        "안구건조증", "익상편", "황반변성"
    ],
    "이비인후과/청각 질환": [
        "난청", "메니에르병", "비염", "이명", "중이염", "코골이", "코막힘"
    ],
    "비뇨기과/생식기 질환": [
        "골반염", "남성갱년기", "다낭성난소증후군(PCOS)", "사정장애", "생리불순",
        "생리통", "성병(임질, 매독)", "신장결석", "여성갱년기", "요로결석",
        "요실금", "자궁근종", "전립선비대증", "질염", "폐경증후군"
    ],
    "치과질환": [
        "구내염", "이갈이", "입냄새(구취)", "충치(치아우식증)", "치은염",
        "치주염(잇몸병)", "턱관절 장애"
    ],
    "종양 및 암": [
        "간암", "갑상선암", "담낭암", "대장암", "림프종", "방광암", "백혈병",
        "신장암", "위암", "유방암", "자궁경부암", "전립선암", "췌도암", "췌장암",
        "폐암", "피부암"
    ],
    "기타 질환": [
        "과호흡증후군", "냉증", "부종", "빈혈", "열사병", "저혈압"
    ]
}

# 입력 및 출력 디렉토리
input_dir = "csv/predictions" # 예측 결과 CSV 파일들이 있는 위치
output_dir = "csv/graphs" # 카테고리별 집계 CSV 파일을 저장할 위치
os.makedirs(output_dir, exist_ok=True) # 출력 디렉토리가 존재하지 않은 경우 생성

# 카테고리별로 순회 (category: 카테고리 이름, disease_list: 해당 카테고리의 질병 리스트)
for category, disease_list in category_disease_map.items():
    records = [] # 해당 카테고리에 속한 질병들의 결과를 담을 리스트 초기화
    
    # 해당 카테고리에 포함된 각 질병에 대해 반복
    for disease in disease_list:
        # 예측 결과 CSV 파일 이름 구성 (예: predictions_disease_top3_감기.csv)
        filename = f"predictions_disease_top3_{disease}.csv"
        
        # 전체 경로 구성 (입력 디렉토리 + 파일명)
        filepath = os.path.join(input_dir, filename)
        
        # 파일이 존재하지 않으면 해당 질병은 건너뜀
        if not os.path.exists(filepath):
            print(f"❌ {filename}은 없으므로 건너뜀")
            continue
        
        # CSV 파일을 읽어와서 DataFrame으로 로드
        df = pd.read_csv(filepath)
        
        # 샘플 수 계산 (해당 질병에 대해 예측을 수행한 총 문장 수)
        num_samples = len(df)
        
        # Top-k accuracy 계산: topk_accuracy 컬럼의 합계를 샘플 수로 나누어 평균 정확도 도출
        topk_accuracy = df["topk_accuracy"].sum() / num_samples
        
        # MRR(Mean Reciprocal Rank) 계산: mrr_score 컬럼의 평균
        mrr_score = df["mrr_score"].sum() / num_samples
        
        # 평균 rank 계산
        avg_rank = df["rank"].replace(0, pd.NA).dropna().mean()

        # 계산된 예측 성능 지표들을 하나의 딕셔너리로 정리하여 records 리스트에 추가
        records.append({
            "category": category, # 현재 질병이 속한 카테고리명
            "disease": disease, # 질병명
            "topk_accuracy": round(topk_accuracy, 4), # 소수점 4자리까지 반올림한 Top-k 정확도
            "mrr_score": round(mrr_score, 4), # 소수점 4자리까지 반올림한 MRR
            "avg_rank": round(avg_rank, 2) # 소수점 2자리까지 반올림한 평균 랭크
                if pd.notna(avg_rank) else None # 평균 랭크가 NaN이면 None으로 처리
        })
    
    # 만약 현재 카테고리에서 처리된 데이터가 존재할 경우   
    if records:
        # records 리스트를 DataFrame으로 변환
        output_df = pd.DataFrame(records)
        
        # 저장 경로를 카테고리명을 기반으로 설정
        output_path = os.path.join(output_dir, f"{category}.csv")
        try:
            # DataFrame을 CSV 파일로 저장 (UTF-8-sig 인코딩으로 한글 호환)
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"✅저장 완료: {output_path}")
        except Exception as e:
            # 저장 중 에러 발생 시 에러 메시지 출력
            print(f"❌ 저장 실패: {output_path} — {e}")
    else:
        # 처리된 데이터가 없을 경우 경고 메시지 출력
        print(f"⚠️ {category}: 저장할 내용 없음")