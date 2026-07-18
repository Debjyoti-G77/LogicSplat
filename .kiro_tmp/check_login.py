from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.whoami()
    print(f"Logged in as: {info['name']}")
except Exception as e:
    print(f"Not logged in: {e}")
