import datetime
from firebase_admin import db
from DB.DBconfig import reader_test_table, queue_table
from AI.modelingUtil import isRunning,run_model_task, setRunning

def process_tasks(reader_test_id, type):
    # ReaderTest 테이블에서 이미지 정보 가져오기
    reader_test_item = reader_test_table.child(f"{type}/{reader_test_id}").get()
    
    if not reader_test_item:
        return {"error": "ReaderTest ID not found"}

    image_ids = reader_test_item.get('Image', [])
    if not isinstance(image_ids, list):
        return {"error": "Invalid image format in ReaderTest"}

    for image_id in image_ids:
        timestamp = datetime.datetime.now().isoformat()
        task_id = f"task_{reader_test_id}_{image_id}_{int(datetime.datetime.now().timestamp() * 1000)}"
        queue_table.child(task_id).set({
            "task_id": task_id,  # 작업 고유 ID
            "status": "PENDING",  # 작업 상태
            "timestamp": timestamp,  # 작업 생성 시간
            "reader_test_id": reader_test_id,  # 연결된 ReaderTest ID
            "image_id": image_id,  # 작업 이미지 ID
        })

    return {"message": "Tasks added successfully"}
def get_pending_task():
# PENDING 상태의 작업 가져오기+ failed인것도 가져오기
    pending_tasks = queue_table.order_by_child("status").equal_to("PENDING").get()

    # 작업이 없는 경우 반환
    if not pending_tasks:
        # print("No pending tasks found.")
        return

    try:
# 가장 오래된 작업 선택
        oldest_task = min(
            pending_tasks.items(),
            key=lambda item: item[1].get("timestamp", "")
        )
        return oldest_task
    except ValueError:
        print("No valid tasks found in pending_tasks.")
        return
    except KeyError as e:
        print(f"Missing expected key in task data: {e}")
        return
