# 파일 입출력 관련 import
import csv  # CSV 파일 읽기 및 쓰기를 위한 표준 라이브러리

# 문자열 처리 관련 import
import re  # 정규표현식을 통한 문자열 전처리용 라이브러리

# 파일 경로 상수 정의
INPUT_FILE = 'raw_sentences.txt'  # 문장이 저장된 입력 파일 경로
OUTPUT_FILE = 'train_dataset.csv'  # 전처리된 데이터를 저장할 출력 CSV 파일 경로

# -------------------------------------------------- #
# 텍스트 전처리
# -------------------------------------------------- #
def clean_text(
    text: str # 전처리할 원본 문자열
) -> str: # 전처리한 문자열
    return text.replace('"', '').strip() # 큰 따옴표 제거 후 양쪽 공백 제거

# -------------------------------------------------- #
# 텍스트 데이터 파싱 (질병명, 문장, 스타일 번호 추출)
# -------------------------------------------------- #
def parse_line(
    line: str # 쉼표와 따옴표로 구분된 입력 문자열
):
    # 정규 표현식을 이용해 '질병명', "문장", 스타일 형식의 데이터를 매칭
    match = re.match(r'(.+?),"(.*)",(\d+)', line)
    
    if match: # 정규 표현식과 일치하는 문자열일 경우
        disease = clean_text(match.group(1)) # 첫 번째 항목: 질병명
        sentence = clean_text(match.group(2)) # 두 번째 항목: 문장
        style = match.group(3) # 세 번째 항목: 스타일 번호
        return disease, sentence, style # 튜플 형태로 반환

    # 줄 끝의 개행 문자 제거 후 쉼표(,)를 기준으로 분할
    parts = line.strip().split(',')
    
    # 최소한 질병명, 문장, 스타일 번호 세 항목이 존재하는 경우에만 처리
    if len(parts) >= 3:
        disease = clean_text(parts[0]) # 첫 번째 항목: 질병명
        style = clean_text(parts[-1]) # 마지막 항목: 문장
        sentence = clean_text(','.join(parts[1:-1])) # 중간 항목들을 다시 결합하여 문장으로 처리
        return disease, sentence, style # 튜플 형태로 반환

    # 형식이 잘못된 경우, 질병명만 반환하고, 문장과 스타일 번호는 빈 문자열로 처리
    return line.strip(), '', ''  

# -------------------------------------------------- #
# 입력 파일을 읽고, 파싱한 데이터를 CSV 파일로 저장
# -------------------------------------------------- #
def main():
    # 입력 파일을 읽기 모드로 열고, 출력 파일을 쓰기 모드로 동시에 열기
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:

        # CSV 파일에 데이터를 쓸 writer 객체 생성
        writer = csv.writer(outfile)

        for line in infile: # 입력 파일의 각 줄을 순회
            parsed = parse_line(line) # 각 줄을 파싱하여 튜플로 변환
            writer.writerow(parsed) # 변환된 결과를 CSV 한 줄로 저장 

    # 모든 줄 저장 완료 후 출력 파일명을 포함한 완료 메시지 출력
    print(f"✅ 전체 줄 저장 완료: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
