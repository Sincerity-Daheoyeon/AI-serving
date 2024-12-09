from flask import jsonify, request
import requests

from scheduling.loadToDB import process_tasks
COMPLETION_SIGNAL_URL = "https://your-backend-url.com/completion_signal"  # 백엔드 URL

def process_json(request):
    # JSON 파일 수신
    data = request.json
    reader_test_id = data.get('reader_test_id')
    type = data.get('type')
    print(f"Received JSON: {data}")

    if not reader_test_id or not type:
        return jsonify({"error": "Missing ReaderTest ID or Type"}), 400
    # 작업 처리 호출
    result = process_tasks(reader_test_id, type)

    if result.get('error'):
        return jsonify(result), 400

    return jsonify(result), 200 

def send_completion_signal(reader_test_id):
    """
    작업 완료 신호를 백엔드에 전달
    """
    try:
        response = requests.post(COMPLETION_SIGNAL_URL, json={"reader_test_id": reader_test_id})
        if response.status_code == 200:
            print(f"Completion signal sent for reader_test_id: {reader_test_id}")
        else:
            print(f"Failed to send completion signal: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending completion signal for reader_test_id: {reader_test_id} - {e}")