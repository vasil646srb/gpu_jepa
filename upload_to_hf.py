"""
📤 Загрузка весов Text-JEPA на HuggingFace Hub
"""
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

REPO_ID = "vasil646/jepa_text"
CHECKPOINTS_DIR = "./checkpoints"
MODEL_DIR = "./bge-small-en-v1.5-onnx-Q"

def upload_weights():
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, exist_ok=True, private=False)
    print(f"✅ Репозиторий {REPO_ID} готов")
    
    if Path(CHECKPOINTS_DIR).exists() and list(Path(CHECKPOINTS_DIR).glob("*.pt")):
        print(f"\n📤 Загрузка чекпоинтов...")
        upload_folder(folder_path=CHECKPOINTS_DIR, path_in_repo="checkpoints", repo_id=REPO_ID)
    
    if Path(MODEL_DIR).exists() and list(Path(MODEL_DIR).glob("*.onnx")):
        print(f"\n📤 Загрузка базовой модели...")
        upload_folder(folder_path=MODEL_DIR, path_in_repo="bge-small-en-v1.5-onnx-Q", repo_id=REPO_ID)
    
    print(f"\n🎉 Готово! https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_weights()
