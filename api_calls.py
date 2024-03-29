import requests
import json
import os
BASE_URL = os.environ.get('AI_DIARY_URL')
AI_DIARY_BEARER = os.environ.get('AI_DIARY_BEARER')
AI_USER_EMAIL = os.environ.get("AI_USER_EMAIL")
AI_USER_HASH = os.environ.get("AI_USER_HASH")
AI_USER_NAME = os.environ.get("AI_USER_NAME")

def get_ai_user():
    url = os.path.join(BASE_URL, "api/ai_user")

    payload = json.dumps({
      "email": AI_USER_EMAIL,
      "password": AI_USER_HASH,
      "name": AI_USER_NAME
    })
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {AI_DIARY_BEARER}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    return response.json().get('ai_user')
