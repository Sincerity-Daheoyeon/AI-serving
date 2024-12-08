from asyncio import exceptions
import os
import boto3
import firebase_admin
from firebase_admin import credentials, db
 
try:
    # Firebase Admin SDK 초기화
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://capstone-2402-default-rtdb.firebaseio.com"
    })
except exceptions.FirebaseError as e:
    print(f"Firebase initialization error: {e}")
    exit(1)

# Firebase DB 참조
reader_test_table = db.reference("ReaderTest")
queue_table = db.reference("Queue")
patients_table = db.reference("Patients")
output_table = db.reference("Output")

# S3 설정
bucket_name = os.getenv("AWS_S3_BUCKET")


# AWS 인증 정보 (환경 변수에서 가져오기)
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")  # AWS Access Key ID
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")  # AWS Secret Access Key
aws_region = os.getenv("AWS_REGION", "ap-northeast-2")  # AWS 리전 (기본값 설정)

# S3 버킷 이름
bucket_name = os.getenv("AWS_S3_BUCKET", "default-bucket-name")  # 기본값으로 S3 버킷 이름 제공

s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

# DB 참조를 export
__all__ = ["s3", "reader_test_table", "queue_table", "patients_table", "output_table"]
