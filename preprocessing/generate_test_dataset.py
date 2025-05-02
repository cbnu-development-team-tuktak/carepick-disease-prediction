# Requests 및 데이터 핸들링 관련 import
import requests # HTTP 요청을 보내기 위한 라이브러리

import pandas as pd # 데이터프레임(DataFrame) 처리 라이브러리
import time # 요청 시간에 지연을 주기 위한 시간 관련 함수
import re # 정규표현식을 통한 텍스트 정제 처리

# HTML 파싱 관련 import
from bs4 import BeautifulSoup # HTML 문서를 파싱하고 탐색하는 라이브러리

# User-Agent 생성 관련 import
from fake_useragent import UserAgent # 무작위로 다양한 브라우저 User-Agent를 생성하는 라이브러리

# 수집에 필요한 기본 설정값 정의
BASE_URL = "https://kin.naver.com" # 네이버 지식인 검색 도메인
SEARCH_QUERY = "감기인가요?" # 검색에 사용할 키워드
LABEL = "감기" # 수집 데이터에 부여할 라벨명
TARGET_COUNT = 30 # 수집할 질문 개수

ua = UserAgent() # 각 요청에 대해 무작위 User-Agent 헤더를 생성하기 위한 객체

# ------------------------------------------- #
# 1. 검색 결과에서 URL 추출
# ------------------------------------------- #
def get_search_results(
    page # 검색 결과를 가져올 페이지 번호
):
    headers = {"User-Agent": ua.random} # 랜덤한 User-Agent로 요청 헤더 구성
    url = f"https://kin.naver.com/search/list.naver?query={SEARCH_QUERY}&page={page}" # 검색 결과 페이지 생성
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
    url # 질문 상세 페이지 URL
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

    # 답변 내용에 "감기"라는 단어가 포함되지 않으면 제거
    if "감기" not in answer_text:
        return None

    # 유효한 질문-답변 데이터 딕셔너리 형태로 변환
    return {
        "label": LABEL, # 라벨(질병명) 지정
        "text": question_text, # 질문 본문
        "answer": answer_text, # 답변 본문
        "url": url # 질문 URL
    }

def main():
    all_data = [] # 수집된 질문-답변 데이터를 저장할 목록
    page = 1 # 검색할 첫 번째 페이지 번호

    while len(all_data) < TARGET_COUNT: # 목표 개수(TARGET_COUNT)만큼 데이터를 수집할 때까지 반복
        print(f"[INFO] 페이지 {page} 검색 중...") # 현재 수집 중인 페이지 출력
        urls = get_search_results(page) # 해당 페이지에서 질문 URL 목록 추출
        print(f"[INFO] → {len(urls)}개 URL 수집됨") # 수집된 URL 개수 출력
 
        for url in urls: # 수집한 각 질문 URL에 대해 반복
            result = get_question_and_answer(url) # 질문과 답변(전문의 + 질병 언급 여부 확인 포함) 추출
            if result: # 결과가 있는 경우
                all_data.append(result) # 유효한 결과만 리스트에 추가
                print(f"[OK] 수집됨 ({len(all_data)}/{TARGET_COUNT}): {result['text'][:30]}...") # 일부 질문 내용 출력
                if len(all_data) >= TARGET_COUNT: # 목표 개수 도달 시 
                    break # 반복 중단
            time.sleep(1.0) # 서버 요청 과부하 방지를 위한 딜레이 설정

        page += 1 # 다음 페이지로 이동하여 검색 계속

    df = pd.DataFrame(all_data) # 수집한 데이터를 DataFrame으로 반환
    df.to_csv("test_dataset_gamgi.csv", index=False, encoding='utf-8-sig') # CSV 파일로 저장 (UTF-8 인코딩)
    print(f"\n✅ 최종 저장 완료: {len(df)}개 질문 → test_dataset_gamgi.csv") # 저장 완료 메시지 출력

if __name__ == "__main__":
    main()