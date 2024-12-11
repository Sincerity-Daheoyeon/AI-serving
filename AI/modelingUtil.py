from io import BytesIO
from urllib.parse import urlparse

import cv2
import numpy as np
import tensorflow
import torch
from DB.DBconfig import bucket_name, output_table, patients_table, queue_table, s3
from DB.DBconfig import bucket_name
from boto3.dynamodb.conditions import Key

from scheduling.loadToDB import increment_processed_images_and_check

np.random.seed(0)
tensorflow.random.set_seed(0)

def run_model_task(task_data, model):
    """모델 실행 로직"""
    try:
        print(f"Processing task: {task_data}")
        update_task_status(task_data["task_id"], "IN_PROGRESS")

        # Meta 데이터 가져오기
        # patient_meta = get_patient_meta_by_src(task_data['image_id'])
        # if not patient_meta:
        #     print(f"Meta not found for image_id: {task_data['image_id']}")
        #     update_task_status(task_data['task_id'], "FAILED")
        #     return False
        # image_meta = task_data['image_id'].split('/')[0]
        # image_src = task_data['image_id'].split('/')[-1]
        image_src = task_data['image_id']
        png_data = get_png_from_s3( image_src)  # 메모리 내에서 PNG 로드
        # npzFile = png_to_npz(png_data)  # PNG를 NPZ로 변환

        
        # # 이미지 크기 조정 (224x224로 변환)
        # unstructured = np.load(
        #     npzFile,
        #     allow_pickle=True,
        # )['cube']

        # # 데이터 크기 확인
        # print("Initial unstructured shape:", unstructured.shape)

        # 채널 크기가 3인지 확인
        # if unstructured.shape[1] != 3:
        #     raise ValueError(f"Invalid channel size: {unstructured.shape[1]}, expected 3.")
        # 모델에 그냥 png
        '''244,244,3 -> 1,244,244,3 => binary로. [[0,1]]일거라서 [0]이런식으로 따와서 argmax() 최종int값에 대해서 몇번째 class가 가자으다만 넘겨주면 됨'''
        numpyPng = preprocess_image(png_data)
        # Tensor 변환
        unstructured_tensor = torch.tensor(numpyPng, dtype=torch.float32).to(model.device)

        # 모델 호출
        prediction = model(unstructured_tensor)

        # Swin 모델 출력 처리
        logits = prediction.logits  # SwinImageClassifierOutput에서 logits 추출
        logits_np = logits.detach().cpu().numpy()  # NumPy 배열로 변환

        # NaN 값을 0으로 대체
        logits_np = np.nan_to_num(logits_np, nan=0.0)

        
        # logits 배열에서 마지막 샘플 추출
        result = logits_np[-1]  # 마지막 샘플 (배치 크기가 1인 경우 [num_classes])
        print("로짓값!!!!!",logits_np,"리설트@!!@@" ,result)
        # 가장 큰 값의 인덱스 추출
        predicted_class = np.argmax(result)  # 가장 높은 확률의 클래스 인덱스
        predicted_class = int(predicted_class)  # numpy.int64 -> int 변환


        #3d넣는 3d컨볼루젼
        # unstructured = np.asarray([
        #     cv2.resize(
        #         src=unstructured[..., i],
        #         dsize=(224, 224),
        #     ) for i in range(unstructured.shape[-1])
        # ])

        # if unstructured.shape[0] >= 3:
        #     unstructured = unstructured[:3, :, :]
        # else:
        #     # 채널이 부족하면 복제해서 3채널로 확장
        #     unstructured = np.repeat(unstructured, repeats=3 // unstructured.shape[0], axis=0)[:3, :, :]

        # # 차원 변환: (채널, 높이, 너비) -> (배치 크기, 채널, 높이, 너비)
        # unstructured = np.expand_dims(unstructured, axis=0)  # 배치 차원 추가
        # print("!!unstructured.shape!!: ", unstructured.shape)
        # unstructured_tensor = torch.tensor(unstructured, dtype=torch.float32).to(model.device)
        # 모델 호출
        # prediction = model(unstructured_tensor)
        # NaN값만 0으로 대체
        # prediction = np.nan_to_num(prediction, nan=0.0) 

        # # 결과 저장

        # if isinstance(prediction, torch.Tensor):
        #     logits = prediction.detach().cpu().numpy()
        # else:
        #     logits = prediction.logits.detach().cpu().numpy()        
        # logits_np = logits.detach().cpu().numpy()  # NumPy 배열로 변환
        # # Softmax로 확률 계산
        # probabilities = torch.softmax(torch.tensor(logits_np), dim=1).numpy()

        output_table.child(task_data['reader_test_id']).child('tasks').child(task_data['task_id']).set({
            'image_id': image_src,
            'predicted_class': predicted_class,
            'meta': "patient_meta"
        })
        # outputTable에 처리된 task개수 저장 
        # 전체 task가 처리가 끝났는지 확인->끝났으면 back에 신호 수신
        increment_processed_images_and_check(task_data['reader_test_id'])
        
        print(f"Task completed: {task_data}")
        update_task_status(task_data['task_id'], "COMPLETED")
        return True
    except Exception as e:
        print(f"Error processing task {task_data['task_id']}: {str(e)}")
        update_task_status(task_data['task_id'], "FAILED")
        return False
def preprocess_image(src):
    """이미지를 모델 입력 형식으로 변환"""
    image = decode_png(src)
    # 이미지 크기 조정 (224x224)
    resized_image = cv2.resize(image, (224, 224))  # (H, W, C)
    
    # 채널 변환: (H, W, C) -> (C, H, W)
    resized_image = np.transpose(resized_image, (2, 0, 1))  # (C, H, W)
    
    # 배치 차원 추가: (C, H, W) -> (1, C, H, W)
    resized_image = np.expand_dims(resized_image, axis=0)  # (1, C, H, W)
    
    return resized_image

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

def extract_s3_info(s3_url):
    """S3 URL에서 버킷 이름과 키 추출"""
    parsed_url = urlparse(s3_url)
    # 호스트에서 버킷 이름 추출
    bucket_name = parsed_url.netloc.split('.')[0]
    # 경로에서 S3 키 추출
    key = parsed_url.path.lstrip('/')
    return bucket_name, key

def get_png_from_s3( image_src):
    """S3에서 PNG 파일 읽기 (메모리 내 처리)"""
    try:
        # URL에서 S3 키 추출
        bucket_name, image_id = extract_s3_info(image_src)
        print(f"Extracted S3 Key: {image_id}")

        # S3에서 객체 가져오기
        response = s3.get_object(Bucket=bucket_name, Key=image_id)
        print(f"Downloaded {image_id} from S3 bucket: {bucket_name}")

        # PNG 데이터를 메모리에 로드
        image_data = BytesIO(response['Body'].read())
        return image_data

    except s3.exceptions.NoSuchKey:
        print(f"Error: The key {image_id} does not exist in bucket {bucket_name}.")
    except Exception as e:
        print(f"Error retrieving {image_id} from S3: {str(e)}")
    return None
def decode_png(png_data):
    # PNG 데이터를 OpenCV로 디코딩
    np_array = np.frombuffer(png_data.getvalue(), np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # 컬러 이미지로 읽기 (H, W, C)
    if image is None:
        raise ValueError("Failed to decode PNG data.")
    return image
def png_to_npz(png_data):
    """PNG 데이터를 NumPy 배열로 변환하고 메모리 내에서 사용"""
    # PNG 데이터를 OpenCV로 읽기
    np_array = np.frombuffer(png_data.getvalue(), np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)         # 컬러(RGB)로 읽기
    if image is None:
        raise ValueError("Failed to decode PNG data.")

    # 이미지 크기 조정 (224x224)
    resized_image = cv2.resize(image, (224, 224,3))  # 초기 크기를 224x224로 맞춤

    # 모델 입력 형식에 맞게 차원 변환
    resized_image = np.transpose(resized_image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    unstructured = np.expand_dims(resized_image, axis=0)    # (C, H, W) -> (1, C, H, W)

    # .npz 파일을 메모리에 저장
    npz_buffer = BytesIO()
    np.savez(npz_buffer, cube=unstructured)
    npz_buffer.seek(0)  # 스트림의 시작 위치로 이동

    return npz_buffer
# def png_to_npz(png_data):
#     """PNG 데이터를 NumPy 배열로 변환하고 메모리 내에서 사용"""
#    # PNG 데이터를 OpenCV로 읽기
#     np_array = np.frombuffer(png_data.getvalue(), np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
#     image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)     # GRAYSCALE로 읽기
#     if image is None:
#         raise ValueError("Failed to decode PNG data.")

#     # 이미지 크기 조정 (128x128)
#     resized_image = cv2.resize(image, (224, 224))  # 초기 크기를 224x224로 맞춤

#     # 모델 입력 형식에 맞게 차원 추가
#     unstructured = np.expand_dims(resized_image, axis=0)  # (1, 128, 128)
#     unstructured = np.expand_dims(unstructured, axis=-1)  # (1, 128, 128, 1)

#     # .npz 파일을 메모리에 저장
#     npz_buffer = BytesIO()
#     np.savez(npz_buffer, cube=unstructured)
#     npz_buffer.seek(0)  # 스트림의 시작 위치로 이동

#     return npz_buffer