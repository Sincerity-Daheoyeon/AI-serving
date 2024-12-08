import time
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI import run_model  # AI/__init__.py에서 정의된 run_model 함수 가져오기

while True:
    try:
        run_model()  # 함수 호출
    except Exception as e:
        print(f"Error during run_model execution: {e}")
    time.sleep(10)
