from asyncio import exceptions
import os
import boto3
from dotenv import load_dotenv
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

load_dotenv()  # .env 파일 로드

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
aws_s3_bucket = os.getenv("AWS_S3_BUCKET")
aws_console = os.getenv("AWS_CONSOLE")

print("AWS Access Key:", aws_access_key)
print("AWS Region:", aws_region)
print("AWS S3 Bucket:", aws_s3_bucket)
print("AWS Console:", aws_console)
# S3 버킷 이름
bucket_name = os.getenv("AWS_S3_BUCKET", "default-bucket-name")  # 기본값으로 S3 버킷 이름 제공

s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

# # S3 버킷 목록 출력
# try:
#     response = s3.list_buckets()
#     print("S3 Buckets:")
#     for bucket in response['Buckets']:
#         print(f" - {bucket['Name']}")
# except Exception as e:
#     print(f"Error accessing S3: {e}")

# # 특정 버킷의 객체 확인
# try:
#     print(f"Listing objects in bucket {aws_s3_bucket}:")
#     response = s3.list_objects_v2(Bucket=aws_s3_bucket)
#     for obj in response.get('Contents', []):
#         print(f" - {obj['Key']}")
# except Exception as e:
#     print(f"Error accessing objects in bucket {aws_s3_bucket}: {e}")
# DB 참조를 export
__all__ = ["s3", "reader_test_table", "queue_table", "patients_table", "output_table"]
