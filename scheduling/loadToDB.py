import datetime
from firebase_admin import db
from DB.DBconfig import reader_test_table, queue_table
from AI.modeling import isRunning,run_model_task, setRunning
def process_tasks(reader_test_id, type):
    # ReaderTest 테이블에서 이미지 정보 가져오기
    reader_test_item = reader_test_table.child(f"{type}/{reader_test_id}").get()
    
    if not reader_test_item:
        return {"error": "ReaderTest ID not found"}

    image_ids = reader_test_item.get('Image', [])
    if not isinstance(image_ids, list):
        return {"error": "Invalid image format in ReaderTest"}

    # 작업 생성 및 큐 추가
    queue_table = db.reference('Queue')  # Firebase Queue 테이블 참조
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

def run_pending_tasks():
    # 대기 중인 작업을 처리하는 로직 구현
    try:
         # PENDING 상태의 작업 가져오기
        pending_tasks = queue_table.order_by_child("status").equal_to("PENDING").get()

        # 작업이 없는 경우 반환
        if not pending_tasks:
            print("No pending tasks found.")
            return

        try:
    # 가장 오래된 작업 선택
            oldest_task = min(
                pending_tasks.items(),
                key=lambda item: item[1].get("timestamp", "")
            )
            task_id, task_data = oldest_task
        except ValueError:
            print("No valid tasks found in pending_tasks.")
            return
        except KeyError as e:
            print(f"Missing expected key in task data: {e}")
            return
        if not isRunning():
            if not isinstance(task_data, dict):
                print(f"Invalid task_data format: {task_data}")
                return
            required_fields = ["task_id", "image_id"]
            missing_fields = [field for field in required_fields if field not in task_data]
            if missing_fields:
                print(f"Missing fields in task_data: {missing_fields}")
                return
            result = run_model_task(task_data)
            # 상태를 확인하여 COMPLETE일 경우 삭제
            updated_task_data = queue_table.child(task_id).get()
            if result:
                queue_table.child(task_id).delete()
                print(f"Task {task_id} deleted from queue.")
            else:
                print(f"Task {task_id} was not deleted; current status: {updated_task_data.get('status')}")        
        else:
            print("Another task is already running.")
    except Exception as e:
        print(f"Error processing tasks: {e}")

