import threading
import time
import schedule
from DB.DBconfig import output_table, patients_table, queue_table
from schedule import fetch_next_task
from boto3.dynamodb.conditions import Key
ai_working_lock = threading.Lock()
ai_working = False

def isRunning():
    global ai_working
    with ai_working_lock:
        return ai_working
    
def setRunning(status):
    global ai_working
    with ai_working_lock:
        ai_working = status

def run_model_task(task_data):
    """모델 실행 로직"""
    setRunning(True)
    try:
        print(f"Processing task: {task_data}")
        update_task_status(task_data["task_id"], "IN_PROGRESS")

        # Meta 데이터 가져오기
        patient_meta = get_patient_meta_by_src(task_data['image_id'])
        if not patient_meta:
            print(f"Meta not found for image_id: {task_data['image_id']}")
            update_task_status(task_data['task_id'], "FAILED")
            return False

        # 모델 실행 (여기서 모델 호출)
        time.sleep(10)  # 모델 실행 대기 시뮬레이션 5초에서 60초. 모델 받아보고 결정하기

        # Output Table에 저장
        output_table.put_item(Item={
            'task_id': task_data['task_id'],
            'image_id': task_data['image_id'],
            'result': 'Success',  # 예제 결과
            'meta': patient_meta
        })

        print(f"Task completed: {task_data}")
        update_task_status(task_data['task_id'], "COMPLETED")
        return True
    except Exception as e:
        print(f"Error processing task {task_data['task_id']}: {str(e)}")
        update_task_status(task_data['task_id'], "FAILED")
        return False
    finally:
        setRunning(False)
#S3랑 잘 연동됐는지 확인해봐라~!~!!~~!@~@~!@~!@#$!@#$!@#$!@$
def get_patient_meta_by_src(image_src):
    try:
        # GSI를 사용해 src로 항목 검색
        response = patients_table.query(
            IndexName='src-index',  # GSI 이름
            KeyConditionExpression=Key('image.src').eq(image_src)
        )
        
        # 검색 결과에서 첫 번째 항목의 메타정보 추출
        items = response.get('Items', [])
        if not items:
            print(f"No patient found for src: {image_src}")
            return None
        
        patient_meta = items[0].get('meta', {})
        print(f"Patient meta found: {patient_meta}")
        return patient_meta
    except Exception as e:
        print(f"Error retrieving patient meta: {e}")
        return None

def update_task_status(task_id, status):
    """작업 상태 업데이트"""
    try:
        queue_table.child(task_id).update({"status": status})
        print(f"Task {task_id} updated to {status}.")
    except Exception as e:
        print(f"Failed to update task status for {task_id}: {e}")


def task_scheduler():
    """스케줄러: Queue에서 작업을 하나씩 가져와 실행"""
    global ai_working
    if ai_working:
        return  # 현재 작업 중이면 아무 작업도 하지 않음

    task = fetch_next_task()
    if not task:
        print("No pending tasks in the queue.")
        return

    # 작업 실행
    ai_working = True
    update_task_status(task['task_id'], "IN_PROGRESS")
    run_model_task(task)

    
def schedule_runner():
    """스케줄링을 별도 스레드에서 실행"""
    while True:
        schedule.run_pending()
        time.sleep(1)



