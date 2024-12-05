from subprocess import Popen, run
from flask import Flask, request, jsonify
from scheduling.loadToDB import process_tasks
import threading
from scheduling.scheduler import schedule_runner

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_json():
    # JSON 파일 수신
    data = request.json
    reader_test_id = data.get('id')
    type = data.get('type')
    if not reader_test_id or not type:
        return jsonify({"error": "Missing ReaderTest ID or Type"}), 400
    # 작업 처리 호출
    result = process_tasks(reader_test_id, type)
    if result.get('error'):
        return jsonify(result), 400

    return jsonify({"message": "Tasks added to queue"}), 200

if __name__ == '__main__':
    # 스케줄러 스레드 실행
    run(["python", "DB.Dbconfig.py"], check=True)
    run(["python", "scheduling.scheduler.py"], check=True)

    # Flask 서버 실행
    app.run(host='127.0.0.1', port=5000)
