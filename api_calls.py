import requests
import json
import os
BASE_URL = os.environ.get('AI_DIARY_URL')
AI_DIARY_BEARER = os.environ.get('AI_DIARY_BEARER')
AI_USER_EMAIL = os.environ.get("AI_USER_EMAIL")
AI_USER_HASH = os.environ.get("AI_USER_HASH")
AI_USER_NAME = os.environ.get("AI_USER_NAME")


def get_ai_user_id():
    url = os.path.join(BASE_URL, "api/ai_user")

    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {AI_DIARY_BEARER}'
    }
    payload = ""

    response = requests.request("GET", url, headers=headers, data=payload)

    if response.json().get('ai_user') in (None, "does not exist"):
        payload = json.dumps({
          "email": AI_USER_EMAIL,
          "password": AI_USER_HASH,
          "name": AI_USER_NAME
        })

        response = requests.request("POST", url, headers=headers, data=payload)

    return response.json().get('ai_user')


def get_latest_posts_from_diary(latest_post_id):
    url = os.path.join(BASE_URL, f"api/posts/{latest_post_id}")
    payload = {}
    files = {}
    headers = {
      'Authorization': f'Bearer {AI_DIARY_BEARER}'
    }

    response = requests.request("GET", url, headers=headers, data=payload, files=files)

    return response.json().get('new_posts')

