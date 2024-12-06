from subprocess import Popen, run
from flask import Flask, request, jsonify
from Flask.server import process_json
from scheduling.loadToDB import process_tasks
import threading
from scheduling.scheduler import schedule_runner

app = Flask(__name__)

# POST 요청을 처리하는 엔드포인트 설정
@app.route('/process', methods=['POST'])
def handle_process():
    return process_json()

if __name__ == '__main__':
    # 스케줄러 스레드 실행
    run(["python", "DB.Dbconfig.py"], check=True)
    run(["python", "scheduling.scheduler.py"], check=True)

    # Flask 서버 실행
    app.run(host='127.0.0.1', port=5000)
