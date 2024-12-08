import datetime
from DB.DBconfig import output_table, reader_test_table, queue_table

def process_tasks(reader_test_id, type):
    # ReaderTest 테이블에서 이미지 정보 가져오기
    reader_test_item = reader_test_table.child(f"{type}/{reader_test_id}").get()
    
    if not reader_test_item:
        return {"error": "ReaderTest ID not found"}

    image_ids = reader_test_item.get('Image_id', []) #$!!#@$%#$%@%@#@@#@$#@%#@%@#$%
    if not isinstance(image_ids, list):
        return {"error": "Invalid image format in ReaderTest"}
    # 총 이미지 개수
    total_images = len(image_ids)
    if total_images == 0:
        return {"error": "No images found for ReaderTest"}

    # OutputTable 초기화
    output_table.child(reader_test_id).set({
        "id": reader_test_id,
        "total_images": total_images,  # 총 이미지 개수 저장
        "processed_images": 0,        # 모델링 처리된 이미지 개수 초기화
        "tasks": {}                   # 작업 목록 초기화
    })

    # 큐에 작업 추가
    for image_id in image_ids:
        timestamp = datetime.datetime.now().isoformat()
        task_id = f"task_{reader_test_id}_{image_id}_{int(datetime.datetime.now().timestamp() * 1000)}"
        queue_table.child(task_id).set({
            "task_id": task_id,  # 작업 고유 ID #RRQEW@#$!TEYWETYR%^@#YWQ%TYWUEIY$WQ#%^WUIT
            "status": "PENDING",  # 작업 상태
            "timestamp": timestamp,  # 작업 생성 시간
            "reader_test_id": reader_test_id,  # 연결된 ReaderTest ID
            "type": type,
            "image_id": image_id,  # 작업 이미지 ID, src
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

def increment_processed_images_and_check(reader_test_id):
    """
    processed_images 값을 증가시키고, 작업 완료 여부를 확인
    """
    try:
        # 트랜잭션으로 processed_images 증가
        def increment_transaction(current_value):
            if current_value is None:
                return 1  # 값이 없으면 초기화
            return current_value + 1

        # processed_images 값 증가
        new_processed_images = output_table.child(f"{reader_test_id}/processed_images").transaction(increment_transaction)

        # total_images 가져오기
        total_images = output_table.child(f"{reader_test_id}/total_images").get()

        # 완료 여부 확인
        if total_images is not None and new_processed_images == total_images:
            print(f"All tasks completed for reader_test_id: {reader_test_id}")
            notify_completion(reader_test_id)

        return {"message": f"Processed images incremented to {new_processed_images}"}

    except Exception as e:
        print(f"Error incrementing processed images for reader_test_id: {reader_test_id} - {e}")
        return {"error": str(e)}
# scheduling/loadToDB.py
def notify_completion(reader_test_id):
    from Flask.server import send_completion_signal  # 지연 import
    send_completion_signal(reader_test_id)