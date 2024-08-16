import os
import json
import numpy as np
from PIL import Image
import boto3
from segment_anything import SamPredictor, sam_model_registry

def model_fn(model_dir):
    try:
        checkpoint_path = os.path.join(model_dir, 'sam_vit_h_4b8939.pth')
        sam_model = sam_model_registry['default'](checkpoint=checkpoint_path)
        return sam_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            return input_data
        raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        print(f"Error in input_fn: {e}")
        raise

def predict_fn(input_data, model):
    try:
        bucket = input_data['bucket']
        key = input_data['key']
        bounding_box = input_data['bounding_box']

        # S3에서 이미지 다운로드
        s3 = boto3.client('s3')
        download_path = '/tmp/image.jpg'
        s3.download_file(bucket, key, download_path)

        image = Image.open(download_path)
        image_np = np.array(image)

        sam_predictor = SamPredictor(model)

        # 바운딩 박스 좌표 변환
        left = bounding_box['Left'] * image_np.shape[1]
        top = bounding_box['Top'] * image_np.shape[0]
        width = bounding_box['Width'] * image_np.shape[1]
        height = bounding_box['Height'] * image_np.shape[0]

        # numpy 배열로 변환
        box_np = np.array([[left, top, left + width, top + height]])

        # 예측
        sam_predictor.set_image(image_np)
        masks, _, _ = sam_predictor.predict(box=box_np)

        result = {
            'masks': masks.tolist()
        }

        return result
    except Exception as e:
        print(f"Error in predict_fn: {e}")
        raise

def output_fn(prediction, response_content_type):
    try:
        return json.dumps(prediction), response_content_type
    except Exception as e:
        print(f"Error in output_fn: {e}")
        raise
