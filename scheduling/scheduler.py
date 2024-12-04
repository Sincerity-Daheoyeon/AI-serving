from flask import app
import schedule
import time
import threading
from scheduling.loadToDB import run_pending_tasks

stop_event = threading.Event()

def schedule_runner():
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)

# 서버 종료 시 스케줄러 종료
try:
    threading.Thread(target=schedule_runner, daemon=True).start()
    app.run(host='127.0.0.1', port=5000)
finally:
    stop_event.set()