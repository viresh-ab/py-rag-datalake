from onedrive_client import get_token
import requests
import os

DRIVE_ID = os.getenv("DRIVE_ID")

token = get_token()
url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root/children"

res = requests.get(url, headers={
    "Authorization": f"Bearer {token}"
})

print(res.json())
