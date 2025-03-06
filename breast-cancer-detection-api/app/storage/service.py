from fastapi import UploadFile, File
from typing import List
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class StorageService:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_file(self, file: UploadFile) -> str:
        file_location = self.upload_dir / file.filename
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())
        return str(file_location)

    def save_files(self, files: List[UploadFile]) -> List[str]:
        file_locations = []
        for file in files:
            file_locations.append(self.save_file(file))
        return file_locations

    def get_file_path(self, filename: str) -> str:
        return str(self.upload_dir / filename)

storage_service = StorageService(upload_dir=os.path.join(BASE_DIR, "uploads"))