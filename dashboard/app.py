# 기본 파일 및 경로 처리 관련 import
import os # 파일 경로 생성 및 디렉토리 탐색 등 OS-level 작업에 사용

# 데이터 처리 관련 import
import pandas as pd

# Dash 앱 관련 import
import dash # Dash 앱 프레임워크의 메인 객체
from dash import dcc, html # dcc: 대시보드 컴포넌트 (그래트, 탭 등), html: HTML 요소 구성

# 시각화 관련 import (Plotly)
import plotly.graph_objs as go # 그래프 객체 생성용 (Bar, Scatter 등 시각화 요소 구성)

# CSV 파일이 저장된 디렉토리
GRAPH_DIR = "csv/graphs"

# Dash 앱 초기화
app = dash.Dash(__name__)
app.title = "질병 예측 정확도 대시보드"

# CSV 파일 읽어서 탭 생성
def generate_tabs():
    tabs = [] # 각 카테고리별 그래프 탭을 저장할 리스트
    
    # GRAPH_DIR 내의 파일들을 정렬하여 반복
    for csv_file in sorted(os.listdir(GRAPH_DIR)):
        # CSV 파일이 아닌 경우 건너뜀
        if not csv_file.endswith(".csv"):
            continue
        
        # 전체 파일 경로 구성
        file_path = os.path.join(GRAPH_DIR, csv_file)
        
        try:
            df = pd.read_csv(file_path) # CSV 파일을 pandas DataFrame으로 읽기
        except Exception as e:
            # 파일 읽기 중 에러 발생 시 로그 출력
            print(f"❌ 오류 발생: {csv_file} 읽기 실패 - {e}")
            continue
        
        # 확장자(.csv)를 제거하여 카테고리명 추출
        category = os.path.splitext(csv_file)[0]
        
        # Plotly 그래프 객체 초기화
        fig = go.Figure()
        
        # Top-3 Accuracy (막대 차트)
        fig.add_trace(go.Bar(
            x=df["disease"], # X축: 질병 이름
            y=df["topk_accuracy"], # Y축: Top-3 정확도
            name="Top-3 Accuracy", # 범례에 표시될 이름
            yaxis="y1" # 첫 번째 Y축에 연결
        ))
        
        # MRR Score (선 차트, 이중축)
        fig.add_trace(go.Scatter(
            x=df["disease"], # X축: 질병 이름 
            y=df["mrr_score"], # Y축: MRR 점수 
            mode="lines+markers", # 선 + 점 모드로 표시
            name="MRR Score", # 범례에 표시될 이름
            yaxis="y2" # 두 번째 Y축에 연결
        ))
        
        # 그래프의 레이아웃(스타일 및 배치) 설정
        fig.update_layout(
            title=f"{category} - 예측 성능 비교", # 그래프 상단 제목
            xaxis=dict(
                title="질병명", # X축 제목 
                tickangle=45 # X축 라벨을 45도 기울여 표시 
            ),
            yaxis=dict(
                title="Top-3 Accuracy", # 왼쪽 Y축 제목
                side="left", # 왼쪽에 위치
                range=[0, 1]
            ),
            yaxis2=dict(
                title="MRR Score", # 오른쪽 Y축 제목
                overlaying="y", # 첫 번째 Y축 위에서 겹쳐서 표시
                side="right" # 오른쪽에 위치
            ),
            legend=dict(x=0.01, y=0.99), # 범례 위치 (왼쪽 위)
            margin=dict(l=40, r=40, t=60, b=100), # 그래프 주변 여백 설정
            height=600 # 그래프 높이 설정 (px 단위)
        )
        
        # 해당 카테고리에 대한 탭 추가
        tabs.append(
            dcc.Tab(
                label=category, # 탭에 표시될 카테고리 이름
                children=[dcc.Graph(figure=fig)]) # 그래프를 탭의 내용으로 추가
        )
    
    return tabs

# 전체 레이아웃 구성
app.layout = html.Div([
    html.H1("질병 카테고리별 예측 성능 대시보드", style={"textAlign": "center"}),
    dcc.Tabs(children=generate_tabs())
])

if __name__ == "__main__":
    app.run(debug=True)