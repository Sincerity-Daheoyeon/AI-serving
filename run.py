import os
from subprocess import run
import sys
import threading
from flask import Flask, request
from Flask.server import process_json

app = Flask(__name__)
# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# POST 요청을 처리하는 엔드포인트 설정
@app.route('/reader-test', methods=['POST'])
def handle_process():
    return process_json(request)

@app.route('/')
def hello():
    return "Hello, World!"

def run_flask():
    print("서버 시작합니다~~~~~~~~~")
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Flask 서버를 별도의 스레드에서 실행
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
    flask_thread.start()

    # 메인 스레드에서 다른 작업 실행
    print("서버 실행 중... 다른 작업 실행")
    
    # 스케줄러 스레드 실행
    run(["python", "DB/DBconfig.py"], check=True)
    run(["python", "scheduling/scheduler.py"], check=True)