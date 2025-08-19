import kagglehub
import requests

# 设置超时为 30 秒（默认可能只有 5 秒）
# kagglehub.http_client.timeout = 30
# Download latest version
path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")

print("Path to dataset files:", path)