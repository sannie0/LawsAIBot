import requests

promt = {
    "modelUri": "gpt://b1gtm1dktbnprji5rmcp/yandexgpt-lite",
    #"modelUri": "gpt://b1gtm1dktbnprji5rmcp/gpt-model-id",
    "completionOptions":{
        "stream":False,
        "temperature":0.6,
        "maxTokens": 2000
    },
    "messages":[
        {
            "role": "user",
            "text": "Напиши описание идеального отдыха"
        }
     ]
}


'''!!!создать переменную среды через cmd для ключа - set API_KEY=наш_ключ
создать переменную среды через cmd для ключа - set FOLDER_ID=ваш_folder_id
import os

# Получение API-ключа из переменной среды
api_key = os.getenv("API_KEY")

#Получение id из переменной среды
folder_id = os.getenv("FOLDER_ID")

'''

url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {api_key}",
    "x-folder-id": folder_id
}

response = requests.post(url, headers=headers, json=promt)
result = response.text
print(result)









'''
не нужно
# Получение API-ключа из переменной среды
api_key = os.getenv("API_KEY")

#Получение id из переменной среды
folder_id = os.getenv("FOLDER_ID")
'''