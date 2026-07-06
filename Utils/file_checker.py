import os
import hashlib
import json

from Utils.logger import LoggerManager


class FileChecker:

    logger = LoggerManager.get_logs("FileChecker")

    @staticmethod
    def already_exists(output_path: str) -> bool:
        if os.path.exists(output_path):
            FileChecker.logger.info(
                f"File already available."
            )
            return True
        return False

    @staticmethod
    def compute_hash(file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def load_meta(meta_path: str) -> dict:
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_meta(meta_path: str, meta: dict) -> None:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    @staticmethod
    def is_outdated(
        output_path: str,
        dataset_path: str,
        meta_path: str = "Meta/.meta.json"
    ) -> bool:
        if not os.path.exists(output_path):
            return True

        current_hash = FileChecker.compute_hash(dataset_path)
        meta = FileChecker.load_meta(meta_path)
        stored_hash = meta.get(output_path)

        if current_hash != stored_hash:
            FileChecker.logger.info(
                f"Dataset has changed, regenerating: {output_path}"
            )
            return True

        FileChecker.logger.info(
            f"File is up to date, skipping: {output_path}"
        )
        return False

    @staticmethod
    def register(
        output_path: str,
        dataset_path: str,
        meta_path: str = "Meta/.meta.json"
    ) -> None:
        current_hash = FileChecker.compute_hash(dataset_path)
        meta = FileChecker.load_meta(meta_path)
        meta[output_path] = current_hash
        FileChecker.save_meta(meta_path, meta)