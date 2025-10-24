# app/gdrive.py
# Helpers to list/download Google Drive files and export Google Docs.
# Requires: google-api-python-client, google-auth

from __future__ import annotations
import io
import os
from typing import List, Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _get_drive_service(service_account_file: Optional[str] = None):
    service_account_file = service_account_file or os.getenv("SERVICE_ACCOUNT_FILE")
    if not service_account_file or not os.path.exists(service_account_file):
        raise RuntimeError("SERVICE_ACCOUNT_FILE not set or file missing")
    creds = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds)
    return drive

def list_files_in_folder(folder_id: str, service_account_file: Optional[str] = None) -> List[Dict]:
    """
    Returns files in the Drive folder. Each dict contains id, name, mimeType.
    """
    drive = _get_drive_service(service_account_file)
    query = f"'{folder_id}' in parents and trashed = false"
    results = []
    page_token = None
    while True:
        res = drive.files().list(q=query, fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token, pageSize=1000).execute()
        files = res.get("files", [])
        results.extend(files)
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return results

def download_file(file_id: str, dest_path: str, service_account_file: Optional[str] = None):
    """
    Downloads a file from Drive. If the file is a Google Doc, exports as text/plain.
    """
    drive = _get_drive_service(service_account_file)
    # Get metadata to check mime type
    meta = drive.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    mime_type = meta.get("mimeType", "")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    if mime_type == "application/vnd.google-apps.document":
        request = drive.files().export_media(fileId=file_id, mimeType="text/plain")
    else:
        request = drive.files().get_media(fileId=file_id)

    fh = io.FileIO(dest_path, mode="wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()
    return {"path": dest_path, "meta": meta}

def download_folder(folder_id: str, out_dir: str = "data/drive", service_account_file: Optional[str] = None) -> List[Dict]:
    """
    Downloads all files in a Drive folder into out_dir/<fileid>-<name>.txt (or original extension)
    Returns list of metadata dicts with 'id','name','path','mimeType'
    """
    files = list_files_in_folder(folder_id, service_account_file)
    downloaded = []
    for f in files:
        file_id = f["id"]
        name = f["name"]
        # sanitize filename
        safe_name = "".join(c if c.isalnum() or c in " .-_()" else "_" for c in name)
        # Use .txt for exported Google Docs; for others keep original name
        mime = f.get("mimeType", "")
        if mime == "application/vnd.google-apps.document":
            out_path = os.path.join(out_dir, f"{file_id}-{safe_name}.txt")
        else:
            out_path = os.path.join(out_dir, f"{file_id}-{safe_name}")
        meta = download_file(file_id, out_path, service_account_file)
        downloaded.append({"id": file_id, "name": name, "path": out_path, "mimeType": mime})
    return downloaded
