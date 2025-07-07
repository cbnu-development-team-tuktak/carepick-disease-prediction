# Requests 및 데이터 핸들링 관련 import
import requests # HTTP 요청을 보내기 위한 라이브러리
import pandas as pd # 데이터프레임(DataFrame) 처리 라이브러리
import time # 요청 시간에 지연을 주기 위한 시간 관련 함수
import re # 정규표현식을 통한 텍스트 정제 처리

# HTML 파싱 관련 import
from bs4 import BeautifulSoup # HTML 문서를 파싱하고 탐색하는 라이브러리

# User-Agent 생성 관련 import
from fake_useragent import UserAgent # 무작위로 다양한 브라우저 User-Agent를 생성하는 라이브러리

# OS 관련 import
import os # 디렉토리 생성을 위한 os 모듈

# 수집에 필요한 기본 설정값 정의
BASE_URL = "https://kin.naver.com" # 네이버 지식인 검색 도메인
TARGET_COUNT = 30 # 수집할 질문 개수
MAX_PAGE = 99 # 네이버 검색 결과 최대 페이지 수 제한

# 저장 경로 설정
OUTPUT_DIR = "csv/test" # 데이터셋 저장 경로
os.makedirs(OUTPUT_DIR, exist_ok = True) # 디렉토리가 없으면 생성

ua = UserAgent() # 각 요청에 대해 무작위 User-Agent 헤더를 생성하기 위한 객체

# ------------------------------------------- #
# 정규 표현식에 해당하는 Q&A만 필터링
# ------------------------------------------- #
def is_diagnosed_as_disease(
    answer_text, # 답변 텍스트
    label # 질병명
):    
    # label에 해당하는 질병을 부정하거나 제외하는 문장 패턴
    negative_patterns = [
        fr"{label}[는]? ?아닌 ?것 ?같습니다",
        fr"{label}(인 줄 알았지만|가 아니라)",
        fr"단순 ?{label}보다",
        fr"{label}로 ?보이지만",
        fr"{label}와는 ?다르",
        fr"{label} ?는 ?아니고",
    ]

    # 부정 진단 패턴에 해당되면 False 반환
    for pattern in negative_patterns:
        if re.search(pattern, answer_text):
            return False

    # 그 외에는 모두 True 반환
    return True

# ------------------------------------------- #
# 1. 검색 결과에서 URL 추출
# ------------------------------------------- #
def get_search_results(
    search_query, # 검색 키워드
    page # 검색 결과를 가져올 페이지 번호
):
    headers = {"User-Agent": ua.random} # 랜덤한 User-Agent로 요청 헤더 구성
    url = f"https://kin.naver.com/search/list.naver?query={search_query}&page={page}" # 검색 결과 페이지 생성
    response = requests.get(url, headers=headers) # 해당 페이지에 HTTP 요청
    if response.status_code != 200: # 요청 실패 시 에러 출력 후 빈 리스트 반환
        print(f"[ERROR] 페이지 {page} 요청 실패: {response.status_code}") 
        return []

    soup = BeautifulSoup(response.text, "html.parser") # 응답 HTML 파싱
    results = soup.select(".basic1 > li") # 질문 항목 리스트 선택
    urls = [] # 결과 URL 저장 리스트

    for item in results:
        link_tag = item.select_one("dt > a") # 각 질문 항목에서 링크 태그 추출
        if link_tag: # 링크 태그가 존재할 경우
            href = link_tag.get("href") # href 속성 추출
            # 상대 경로일 경우 BASE_URL을 붙여 절대 URL로 변환
            full_url = href if href.startswith("http") else BASE_URL + href 
            urls.append(full_url) # 최종 URL 리스트로 추가

    return urls # 수집된 질문 URL 목록 반환

# ------------------------------------------- #
# HTML 파싱 객체에서 답변자가 전문의인지 여부 확인
# ------------------------------------------- #
def is_expert_answer(
    soup # BeautifulSoup 객체 (답변 HTML 파싱 결과)
):
    # 답변자 정보에서 '전문의' 뱃지를 가진 요소를 선택
    badge = soup.select_one("span.badge.expert_job")
    
    # 해당 뱃지가 존재하고, 텍스트에 '전문의'가 포함되어 있는지 여부를 반환
    return badge is not None and "전문의" in badge.get_text()

# ------------------------------------------- #
# 질문 본문과 '전문의'가 작성한 답변을 추출
# ------------------------------------------- #
def get_question_and_answer(
    url, # 질문 상세 페이지 URL
    label # 질병명    
):
    headers = {"User-Agent": ua.random} # 랜덤한 User-Agent 헤더 생성 (봇 차단 우회)
    response = requests.get(url, headers=headers) # 지정된 URL로 HTTP GET 요청 전송
    
    # 요청 실패 시 에러 메시지를 출력하고 None 반환
    if response.status_code != 200:
        print(f"[ERROR] 본문 요청 실패: {url}")
        return None

    # 응답 HTML을 파싱하여 BeautifulSoup 객체 생성
    soup = BeautifulSoup(response.text, "html.parser")

    # 질문 본문 영역 추출
    question = soup.select_one("div.questionDetail")
    if not question: # 질문 본문이 없는 경우
        return None # None 반환

    # 질문 텍스트에서 불필요한 공백 제거
    question_text = re.sub(r"\s+", " ", question.get_text(separator=" ", strip=True))
    
    answer_box = soup.select_one("div.answerArea") # 답변 영역 추출
    if not answer_box or not is_expert_answer(answer_box): # 전문의가 작성한 답변이 아닌 경우
        return None # None 반환

    answer_text_div = answer_box.select_one("div.answerDetail") # 답변 본문이 들어 있는 div 요소 찾기
    if not answer_text_div: # 답변이 없는 경우
        return None # None 반환

    # 답변 텍스트에서 불필요한 공백 제거
    answer_text = re.sub(r"\s+", " ", answer_text_div.get_text(separator=" ", strip=True))

    # 답변 텍스트에 질병 진단 관련 패턴이 포함되어 있지 않으면 제외
    if not is_diagnosed_as_disease(answer_text, label):
        return None # 테스트 데이터셋으로 부적합 → 제외


    # 정답 질병명을 [MASK]로 마스킹
    masked_question_text = re.sub(
        rf"{re.escape(label)}|{re.escape(label.replace(' ', ''))}",
        "[MASK]",
        question_text,
        flags=re.IGNORECASE
    )

    # 다른 질병명들은 [DISEASE]로 마스킹
    for disease in disease_list:
        if disease.lower() == label.lower():
            continue  # 이미 [MASK]로 처리된 질병은 제외
        masked_question_text = re.sub(
            rf"{re.escape(disease)}|{re.escape(disease.replace(' ', ''))}",
            "[DISEASE]",
            masked_question_text,
            flags=re.IGNORECASE
    )
    
    # 유효한 질문-답변 데이터 딕셔너리 형태로 변환
    return {
        "label": label, # 라벨(질병명) 지정
        "text": masked_question_text, # 질문 본문 (마스킹 처리)
        "answer": answer_text, # 답변 본문 
        "url": url # 질문 URL
    }

# ------------------------------------------- #
# 질병별로 데이터 수집 루프 진행
# ------------------------------------------- #
def collect_for_label(
    label # 라벨(질병명)
):
    search_query = f"{label} 증상" # 검색 키워드 생성
    all_data = [] # 수집된 질문-답변 데이터를 저장할 목록
    page = 1 # 검색할 첫 번째 페이지 번호
    
    while len(all_data) < TARGET_COUNT and page <= MAX_PAGE:
        print(f"[INFO] 페이지 {page} 검색 중...") # 현재 수집 중인 페이지 출력
        urls = get_search_results(search_query, page) # 해당 페이지에서 질문 URL 목록 추출
        print(f"[INFO] → {len(urls)}개 URL 수집됨") # 수집된 URL 개수 출력

        for url in urls: # 수집한 각 질문 URL에 대해 반복
            result = get_question_and_answer(url, label) # 질문과 답변(전문의 + 질병 언급 여부 확인 포함) 추출
            if result: # 결과가 있는 경우
                all_data.append(result) # 유효한 결과만 리스트에 추가

                # 마스킹된 질문 텍스트에서 [MASK] 기준으로 앞뒤 15자씩 슬라이싱
                masked_text = result['text']
                start = masked_text.find("[MASK]") # [MASK] 위치 탐색
                if start != -1:
                    preview_start = max(start - 15, 0) # 시작 인덱스 (0 이상)
                    preview_end = start + len("[MASK]") + 15 # 끝 인덱스
                    preview = masked_text[preview_start:preview_end]
                else:
                    preview = masked_text[:30] # [MASK]가 없는 경우 텍스트 앞부분 출력

                print(f"[OK] 수집됨 ({len(all_data)}/{TARGET_COUNT}): {preview}...") # 질문 미리보기 출력

                if len(all_data) >= TARGET_COUNT: # 목표 개수 도달 시
                    break # for 루프 중단

            time.sleep(1.0) # 서버 요청 과부하 방지를 위한 딜레이 설정

        if len(all_data) >= TARGET_COUNT: # for 루프 종료 이후에도 다시 한 번 체크
            break # while 루프까지 종료

        page += 1 # 다음 페이지로 이동

    df = pd.DataFrame(all_data) # 수집한 데이터를 DataFrame으로 반환
    filename = os.path.join(OUTPUT_DIR, f"test_dataset_{label}.csv") # 저장 파일명은 각 질병명이 나타나도록 함
    df.to_csv(filename, index=False, encoding='utf-8-sig') # CSV 파일로 저장 (UTF-8 인코딩)
    print(f"\n✅ ({label}) 저장 완료 → {filename} ({len(df)}개)\n") # 저장 완료 메시지 출력
    

with open("csv/disease_list.csv", encoding="utf-8-sig") as f:
    disease_list = [line.strip() for line in f if line.strip()]
    
if __name__ == "__main__":
    disease_df = pd.read_csv("csv/disease_list.csv") # 질병 목록 로드
    for idx, label in enumerate(disease_df["질병명"]):
        if idx < 77:  # 0부터 시작이므로 39 == 40번째 줄 (경추통)
            continue  # 앞부분은 건너뜀
        
        label = label.strip('"')  # 큰따옴표 제거
        
        filename = os.path.join(OUTPUT_DIR, f"test_dataset_{label}.csv")
        if os.path.exists(filename):
            print(f"[SKIP] {label} → 이미 수집됨")
            continue  # 이미 수집된 경우 건너뜀
        try:
            collect_for_label(label) # 수집 실행
        except Exception as e:
            print(f"[ERROR] {label} 처리 중 오류 발생: {e}")
            time.sleep(10)