from dataclasses import asdict
from typing import Dict, Optional
import requests

from helm.common.cache import Cache, CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.hierarchical_logger import hlog
from helm.common.file_upload_request import FileUploadRequest, FileUploadResult


class GCSClientError(Exception):
    pass


class GCSClient:
    """
    Uploads files to GCS. Ensure the GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
    environment variable is set.
    """

    MAX_CHECK_ATTEMPTS: int = 10

    def __init__(self, bucket_name: str, cache_config: CacheConfig):
        try:
            from google.cloud import storage  # type: ignore
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        self._bucket_name: str = bucket_name
        self._cache = Cache(cache_config)
        self._storage_client: Optional[storage.Client] = None

    def upload(self, request: FileUploadRequest) -> FileUploadResult:
        """Uploads a file to GCS."""
        try:
            from google.cloud import storage  # type: ignore
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        try:

            def do_it():
                if self._storage_client is None:
                    self._storage_client = storage.Client()

                bucket = self._storage_client.bucket(self._bucket_name)
                file_path: str = request.path
                blob = bucket.blob(file_path)

                # Optional: set a generation-match precondition to avoid potential race conditions
                # and data corruptions. The request to upload is aborted if the object's
                # generation number does not match your precondition. For a destination
                # object that does not yet exist, set the if_generation_match precondition to 0.
                # If the destination object already exists in your bucket, set instead a
                # generation-match precondition using its generation number.
                generation_match_precondition: int = 0

                blob.upload_from_filename(file_path, if_generation_match=generation_match_precondition)
                url: str = self._get_url(file_path)

                # Ensure the file was uploaded successfully
                uploaded: bool = False
                for _ in range(0, self.MAX_CHECK_ATTEMPTS):
                    check_response = requests.head(url)
                    if check_response.status_code == 200:
                        uploaded = True
                        break
                assert uploaded, f"File {file_path} was not uploaded successfully."

                hlog(f"File {file_path} uploaded and is available at {url}.")
                return {"url": url}

            cache_key: Dict = asdict(request)
            result, cached = self._cache.get(cache_key, do_it)

        except Exception as e:
            raise GCSClientError(e)

        return FileUploadResult(success=True, cached=cached, url=result["url"])

    def _get_url(self, path: str) -> str:
        return f"https://storage.googleapis.com/{self._bucket_name}/{path}"
