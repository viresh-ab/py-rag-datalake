import os
import requests
import msal
from dotenv import load_dotenv
from pathlib import Path

# Load .env
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
DRIVE_ID = os.getenv("DRIVE_ID")

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["https://graph.microsoft.com/.default"]

def get_token():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET
    )
    token = app.acquire_token_for_client(scopes=SCOPES)
    return token["access_token"]

def get_root_children():
    token = get_token()
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root/children"
    res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    res.raise_for_status()
    return res.json()["value"]

def get_folder_id_by_name(name):
    for item in get_root_children():
        if item["name"] == name and "folder" in item:
            return item["id"]
    raise ValueError(f"Folder '{name}' not found")

def list_pdfs_from_folder_id(folder_id):
    token = get_token()
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{folder_id}/children"
    res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    res.raise_for_status()

    return [
        item for item in res.json()["value"]
        if item.get("file", {}).get("mimeType") == "application/pdf"
    ]

def download_pdf(item_id):
    token = get_token()
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{item_id}/content"
    res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    res.raise_for_status()
    return res.content
