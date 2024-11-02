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

url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Api-Key AQVNxzCOvKxKywDPo_DjEsDADV_40iHXi-_-7aID",
    "x-folder-id": "b1gtm1dktbnprji5rmcp"
}

response = requests.post(url, headers=headers, json=promt)
result = response.text
print(result)
