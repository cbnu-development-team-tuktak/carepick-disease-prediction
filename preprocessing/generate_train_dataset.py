import csv
import re

# 🔧 입력 파일 경로 (텍스트 원본)
INPUT_FILE = 'raw_sentences.txt'

# 📄 출력 CSV 파일 경로
OUTPUT_FILE = 'train_dataset.csv'

# ⚠️ 파싱 실패했을 때 저장할 백업 파일 (현재는 사용 안 함)
# SKIPPED_FILE = 'skipped_lines.txt'

def clean_text(text: str) -> str:
    """큰따옴표를 제거하고 앞뒤 공백 제거"""
    return text.replace('"', '').strip()

def parse_line(line: str):
    """정규식 파싱 시도, 실패하면 대체 파싱 방식 적용"""
    # 1차 시도: 제대로 된 형식
    match = re.match(r'(.+?),"(.*)",(\d+)', line)
    if match:
        disease = clean_text(match.group(1))
        sentence = clean_text(match.group(2))
        style = match.group(3)
        return disease, sentence, style

    # 2차 시도: 큰따옴표 누락되었거나 콤마만 있는 경우
    parts = line.strip().split(',')
    if len(parts) >= 3:
        disease = clean_text(parts[0])
        style = clean_text(parts[-1])
        sentence = clean_text(','.join(parts[1:-1]))
        return disease, sentence, style

    # 3차 시도: 그냥 전부 넣기
    return line.strip(), '', ''  # 저장은 되게 하되, 비정상 줄은 구분 가능

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:

        writer = csv.writer(outfile)

        for line in infile:
            parsed = parse_line(line)
            writer.writerow(parsed)  # 무조건 저장

    print(f"✅ 전체 줄 저장 완료: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
