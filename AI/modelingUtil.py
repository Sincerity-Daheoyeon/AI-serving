from io import BytesIO
import time
import boto3
import cv2
import numpy as np
import schedule
import tensorflow
from DB.DBconfig import bucket_name, output_table, patients_table, queue_table
from schedule import fetch_next_task
from boto3.dynamodb.conditions import Key
from tensorflow.keras.models import load_model

np.random.seed(0)
tensorflow.random.set_seed(0)

def run_model_task(task_data, model):
    """모델 실행 로직"""
    try:
        print(f"Processing task: {task_data}")
        update_task_status(task_data["task_id"], "IN_PROGRESS")

        # Meta 데이터 가져오기
        patient_meta = get_patient_meta_by_src(task_data['image_id'])
        if not patient_meta:
            print(f"Meta not found for image_id: {task_data['image_id']}")
            update_task_status(task_data['task_id'], "FAILED")
            return False
        # 데이터 전처리
        BMI = patient_meta['height'] / (patient_meta['weight'] ** 2)

        structured = np.asarray([
            float(patient_meta['age']),
            0. if patient_meta['sex'] == 'male' else 1.,
            float(BMI),
        ])
        structured = np.expand_dims(structured, axis=0)

        # load .npz
        image_src = task_data['image_id']
        png_data = get_png_from_s3(bucket_name, image_src)  # 메모리 내에서 PNG 로드
        npzFile = png_to_npz(png_data)  # PNG를 NPZ로 변환

        unstructured = np.load(
            npzFile,
            allow_pickle=True,
        )['cube']

        # print("!!!", unstructured.shape)
        unstructured = np.asarray([
            cv2.resize(
                src=unstructured[..., i],
                dsize=(128, 128),
            ) for i in range(unstructured.shape[-1])
        ])

        if unstructured.shape[0] > 500:
            unstructured = np.asarray(
                [
                    unstructured[i] for i in range(0, unstructured.shape[0], 2)
                ]
            )

        unstructured = np.expand_dims(unstructured, axis=-1)
        unstructured = np.expand_dims(unstructured, axis=0)
        # print("!!!", unstructured.shape)
        # 모델 실행 (여기서 모델 호출)
        time.sleep(10)  # 모델 실행 대기 시뮬레이션 5초에서 60초. 모델 받아보고 결정하기
        model.load_weight("")
            # 예시: 모델 입력 데이터 준비
            # 모델 호출
        prediction = model.predict([structured, unstructured],)[0]
        # check is nan => clip
        if np.isnan(prediction).any():
            prediction = np.zeros_like(prediction)
        result = np.argmax(
                    [
                        prediction[0] + prediction[1],
                        prediction[2] + prediction[3],
                        ]
                )
        '''
        num_class = info['num_class']

        if num_class == 2:
            result = np.argmax(
                [
                    prd[0] + prd[1],
                    prd[2] + prd[3],
                    ]
            )

        elif num_class == 4:
            result = np.argmax(prd)

        else:
                raise ValueError('num_class must be 2 or 4')
'''
            # 결과 저장
        output_table.child(task_data['reader_test_id']).update({
            task_data['task_id']: {
                'image_id': task_data['image_id'],
                'result': result,
                'meta': patient_meta
            }
        })

        print(f"Task completed: {task_data}")
        update_task_status(task_data['task_id'], "COMPLETED")
        return True
    except Exception as e:
        print(f"Error processing task {task_data['task_id']}: {str(e)}")
        update_task_status(task_data['task_id'], "FAILED")
        return False
   
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

def get_png_from_s3(bucket_name, image_src):
    """S3에서 PNG 파일 읽기 (메모리 내 처리)"""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=image_src)
    print(f"Downloaded {image_src} from S3")
    
    # PNG 파일 데이터를 메모리로 로드
    image_data = BytesIO(response['Body'].read())
    return image_data

def png_to_npz(png_data):
    """PNG 데이터를 NumPy 배열로 변환하고 메모리 내에서 사용"""
   # PNG 데이터를 OpenCV로 읽기
    np_array = np.frombuffer(png_data.getvalue(), np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
    image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)     # GRAYSCALE로 읽기
    if image is None:
        raise ValueError("Failed to decode PNG data.")

    # 이미지 크기 조정 (128x128)
    resized_image = cv2.resize(image, (128, 128))  # 모델 입력 형식으로 크기 조정

    # 모델 입력 형식에 맞게 차원 추가
    unstructured = np.expand_dims(resized_image, axis=0)  # (1, 128, 128)
    unstructured = np.expand_dims(unstructured, axis=-1)  # (1, 128, 128, 1)

    # .npz 파일을 메모리에 저장
    npz_buffer = BytesIO()
    np.savez(npz_buffer, cube=unstructured)
    npz_buffer.seek(0)  # 스트림의 시작 위치로 이동

    return npz_buffer