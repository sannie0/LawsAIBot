# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# import re
# from langchain_community.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma
# from langchain_community.embeddings.yandex import YandexGPTEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma
# import requests, json
#
#
# # def split_documents(documents: list[Document]):
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=800,
# #         chunk_overlap=80,
# #         length_function=len,
# #         is_separator_regex=False,
# #     )
# #     return text_splitter.split_documents(documents)
#
#
#
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# # def get_embedding_function():
# #     embeddings = BedrockEmbeddings(
# #         credentials_profile_name="default", region_name="us-east-1"
# #     )
# #     return embeddings
#
#
#
#
#
#
# def load_documents(path):
#     document_loader = PyPDFLoader(path)
#     return document_loader.load()
#
#
# # source_file = "C:/Users/Vladislav Yavorskiy/Desktop/hakatonAI/constitution.pdf"
#
#
#
# # def split_documents(documents: list[Document]):
# #     min_chunk_size=300
# #     max_chunk_size=600
# #     max_sentence_size=700
# #     overlap_size=2
# #
# #
# #     all_text = " ".join(doc.page_content for doc in documents)
# #     text_without_metadata = re.sub(r"Document\(metadata=\{.*?\}, page_content='", '', all_text)
# #     cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text_without_metadata)
# #     sentences = re.split(r'(?<=[.!?;])\s+', cleaned_text)
# #
# #     chunks = []
# #     current_chunk = []
# #     current_length = 0
# #
# #     for sentence in sentences:
# #         sentence_length = len(sentence)
# #
# #         # Если предложение превышает максимальную длину, оно добавляется в отдельный чанк
# #         if sentence_length > max_sentence_size:
# #             if current_chunk:
# #                 chunks.append(current_chunk)
# #                 current_chunk = []
# #             chunks.append([sentence])
# #             current_length = 0
# #             continue
# #
# #         # Проверяем, превышает ли чанк лимит, если добавить текущее предложение
# #         if current_length + sentence_length > max_chunk_size and len(
# #                 current_chunk) >= 2 and current_length >= min_chunk_size:
# #             # Добавляем текущий чанк в список и начинаем новый, не добавляя предложение
# #             chunks.append(current_chunk)
# #             current_chunk = []
# #             current_length = 0
# #
# #         # Добавляем предложение в текущий чанк
# #         current_chunk.append(sentence)
# #         current_length += sentence_length
# #
# #         # Если длина чанка достигла допустимого максимума
# #         if current_length >= max_chunk_size:
# #             # Убедимся, что в чанке как минимум два предложения
# #             if len(current_chunk) < overlap_size:
# #                 continue
# #
# #             # Если последнее предложение короткое, добавляем его в следующий чанк
# #             if len(current_chunk[-1]) < 500:
# #                 overlap = current_chunk[-overlap_size:]
# #                 chunks.append(current_chunk)
# #                 current_chunk = overlap
# #                 current_length = sum(len(s) for s in overlap)
# #             else:
# #                 chunks.append(current_chunk)
# #                 current_chunk = []
# #                 current_length = 0
# #
# #         # Добавляем оставшийся текст в последний чанк
# #     if current_chunk:
# #         chunks.append(current_chunk)
# #
# #         # Форматируем чанки
# #     formatted_chunks = [' '.join(chunk) for chunk in chunks if len(' '.join(chunk)) >= min_chunk_size]
# #
# #     return formatted_chunks
#
#
#
# # umber of existing documents in DB: 199
# # Adding new documents: 79
#
#
#
# def get_embedding_function():
#     embeddings = YandexGPTEmbeddings(
#         folder_id = "b1gp04edkv24eujvp7qv",
#         api_key='AQVNwXV4Rl4QNrTw5Kqnbgy7fv3eHyMW6oyit8iw')
#
#     return embeddings
#
#
# def split_documents(documents: list[Document], source: str):
#     min_chunk_size = 800
#     max_chunk_size = 1000
#     max_sentence_size = 700
#     overlap_size = 2
#
#     all_text = []
#     for doc in documents:
#         page_text = doc.page_content
#         page_number = doc.metadata.get('page', 1)  # Предполагается, что метаданные содержат номер страницы
#         cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
#         sentences = re.split(r'(?<=[.!?;])\s+', cleaned_text)
#
#         # Добавляем предложения с номерами страниц
#         for sentence in sentences:
#             all_text.append((sentence, page_number))
#
#     chunks = []
#     current_chunk = []
#     current_length = 0
#     current_page = None
#
#     for sentence, page in all_text:
#         sentence_length = len(sentence)
#
#         # Если предложение превышает максимальную длину, оно добавляется в отдельный чанк
#         if sentence_length > max_sentence_size:
#             if current_chunk:
#                 chunks.append((current_chunk, current_page))
#                 current_chunk = []
#             chunks.append(([sentence], page))
#             current_length = 0
#             current_page = page
#             continue
#
#         # Проверяем, превышает ли чанк лимит, если добавить текущее предложение
#         if current_length + sentence_length > max_chunk_size and len(
#                 current_chunk) >= 2 and current_length >= min_chunk_size:
#             # Добавляем текущий чанк в список и начинаем новый, не добавляя предложение
#             chunks.append((current_chunk, current_page))
#             current_chunk = []
#             current_length = 0
#             current_page = page
#
#         # Если чанк пустой, сохраняем номер страницы для него
#         if not current_chunk:
#             current_page = page
#
#         # Добавляем предложение в текущий чанк
#         current_chunk.append(sentence)
#         current_length += sentence_length
#
#         # Если длина чанка достигла допустимого максимума
#         if current_length >= max_chunk_size:
#             if len(current_chunk) < overlap_size:
#                 continue
#
#             # Если последнее предложение короткое, добавляем его в следующий чанк
#             if len(current_chunk[-1]) < 500:
#                 overlap = current_chunk[-overlap_size:]
#                 chunks.append((current_chunk, current_page))
#                 current_chunk = overlap
#                 current_page = page
#                 current_length = sum(len(s) for s in overlap)
#             else:
#                 chunks.append((current_chunk, current_page))
#                 current_chunk = []
#                 current_length = 0
#
#     # Добавляем оставшийся текст в последний чанк
#     if current_chunk:
#         chunks.append((current_chunk, current_page))
#
#     # Форматируем чанки с метаданными
#     formatted_chunks = [{'content': ' '.join(chunk), 'source': source, 'page': page}
#                         for chunk, page in chunks ]
#     # if len(' '.join(chunk)) >= min_chunk_size
#
#
#     return formatted_chunks
#
#
#
#
# #
# # контекст бота


# # разделенение текста


# # логи в файлик
# # сохранять данные
#
#
#
# CHROMA_PATH = "chroma"
#
#
# def calculate_chunk_ids(chunks):
#     last_page_id = None
#     current_chunk_index = 0
#
#     for chunk in chunks:
#         # Check if chunk is a dictionary and access metadata directly
#         if isinstance(chunk, dict):
#             source = chunk.get("source")
#             page = chunk.get("page")
#             if "metadata" not in chunk:
#                 chunk["metadata"] = {}  # Initialize metadata if not present
#         else:
#             source = chunk.metadata.get("source")
#             page = chunk.metadata.get("page")
#
#         current_page_id = f"{source}:{page}"
#
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0
#
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id
#
#         if isinstance(chunk, dict):
#             chunk["metadata"]["id"] = chunk_id
#         else:
#             chunk.metadata["id"] = chunk_id
#
#     return chunks
#
#
# CHROMA_PATH = "chroma"
#
#
#
# def add_to_chroma(chunks: list):
#     embedding_function = get_embedding_function()
#
#     db = Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=embedding_function
#     )
#
#     # Assign IDs to chunks
#     chunks_with_ids = calculate_chunk_ids(chunks)
#
#     existing_items = db.get(include=[])
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")
#
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         # Access the ID and convert chunk to Document if it’s a dictionary
#         chunk_id = chunk.get("metadata", {}).get("id") if isinstance(chunk, dict) else chunk.metadata.get("id")
#         if chunk_id not in existing_ids:
#             # Convert chunk to Document if it's a dictionary
#             if isinstance(chunk, dict):
#                 chunk = Document(
#                     page_content=chunk.get("content", ""),  # Get content or provide a default empty string
#                     metadata=chunk.get("metadata", {})
#                 )
#             new_chunks.append(chunk)
#
#     if new_chunks:
#         print(f"Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#     else:
#         print("No new documents to add")
#
#
#
#
#
# # AQVN0A4joVdAjT2FC-8kXP4a8Tmp3bf5buywhC0X
#
#
# # AQVNwXV4Rl4QNrTw5Kqnbgy7fv3eHyMW6oyit8iw
# # aje1els4cngg6qqf1p5a
#
#
#
# from langchain.prompts import ChatPromptTemplate
#
# def query_rag(query_text: str):
#     embedding_function = get_embedding_function()
#     db = Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=embedding_function
#     )
#
#     text_prompt = """
#     Ответьте на вопрос, основываясь только на следующем контексте:
#     {context}
#
#     Ответьте на вопрос, основываясь на приведенном выше контексте:
#     {question}
#     """
#
#     results = db.similarity_search_with_score(query_text, k=182)
#
#     for i in results:
#         print(i)
#     print()
#     print()
#     if len(results) == 0 or results[0][1] > 1.25:
#         return "Информация по данному вопросу отсутствует во входных данных."
#
#     context_text = "\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(text_prompt)
#     prompt = prompt_template.format(context = context_text, question = query_text)
#
#
#
#     promt = {
#         "modelUri": "gpt://b1gtm1dktbnprji5rmcp/yandexgpt-lite",
#         "completionOptions": {
#             "stream": False,
#             "temperature": 0.6,
#             "maxTokens": 2000
#         },
#         "messages": [
#             {
#                 "role": "user",
#                 "text": prompt
#             }
#         ]
#     }
#     url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Api-Key AQVNxzCOvKxKywDPo_DjEsDADV_40iHXi-_-7aID",
#         "x-folder-id": "b1gtm1dktbnprji5rmcp"
#     }
#
#     response = requests.post(url, headers=headers, json=promt)
#     data = json.loads(response.text)
#
#     return (data['result']['alternatives'][0]['message']['text'])
#
#
#
#
#
#
#
#
#
#
# #
# # documents = load_documents()
# # chunks = split_documents(documents, source_file)
# #
# #
# #
# # # sorted_chunks = sorted(chunks, key=len, reverse=True)
# # #
# # # for chunk in sorted_chunks:
# # #     print((chunk))
# # #     print()
# #
# #
# # add_to_chroma(chunks)
# # # query_text = "На каком языке осуществляется служебное делопроизводство в РФ?"
# # query_text = "Какой состав Федерального Собрания?"
# #
# # # input("Введите запрос: ")
# # query_rag(query_text)
# #
# # # На каком языке осуществляется служебное делопроизводство в РФ?
# #
#
# # chunks = split_documents(documents)
# # documents = load_documents()
# # print(chunks[0])


from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
import re
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
import requests, json


# source_file = "C:/Users/Vladislav Yavorskiy/Desktop/hakatonAI/Zakon-o-Svyazi-1.pdf"

def load_documents(source_file):
    document_loader = PyPDFLoader(source_file)
    return document_loader.load()



def split_documents(documents: list[Document], source: str):
    min_chunk_size = 300
    max_chunk_size = 600
    max_sentence_size = 700
    overlap_size = 2

    all_text = []
    for doc in documents:
        page_text = doc.page_content
        page_number = doc.metadata.get('page', 1)
        cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
        sentences = re.split(r'(?<=[.!?;])\s+', cleaned_text)

        # Добавляем предложения с номерами страниц
        for sentence in sentences:
            all_text.append((sentence, page_number))

    chunks = []
    current_chunk = []
    current_length = 0
    current_page = None

    for sentence, page in all_text:
        sentence_length = len(sentence)

        # Если предложение превышает максимальную длину, оно добавляется в отдельный чанк
        if sentence_length > max_sentence_size:
            if current_chunk:
                chunks.append((current_chunk, current_page))
                current_chunk = []
            chunks.append(([sentence], page))
            current_length = 0
            current_page = page
            continue

        # Проверяем, превышает ли чанк лимит, если добавить текущее предложение
        if current_length + sentence_length > max_chunk_size and len(
                current_chunk) >= 2 and current_length >= min_chunk_size:
            # Добавляем текущий чанк в список и начинаем новый, не добавляя предложение
            chunks.append((current_chunk, current_page))
            current_chunk = []
            current_length = 0
            current_page = page

        # Если чанк пустой, сохраняем номер страницы для него
        if not current_chunk:
            current_page = page

        # Добавляем предложение в текущий чанк
        current_chunk.append(sentence)
        current_length += sentence_length

        # Если длина чанка достигла допустимого максимума
        if current_length >= max_chunk_size:
            if len(current_chunk) < overlap_size:
                continue

            # Если последнее предложение короткое, добавляем его в следующий чанк
            if len(current_chunk[-1]) < 500:
                overlap = current_chunk[-overlap_size:]
                chunks.append((current_chunk, current_page))
                current_chunk = overlap
                current_page = page
                current_length = sum(len(s) for s in overlap)
            else:
                chunks.append((current_chunk, current_page))
                current_chunk = []
                current_length = 0

    # Добавляем оставшийся текст в последний чанк
    if current_chunk:
        chunks.append((current_chunk, current_page))

    # Форматируем чанки с метаданными
    formatted_chunks = [{'content': ' '.join(chunk), 'source': source, 'page': page}
                        for chunk, page in chunks if len(' '.join(chunk)) >= min_chunk_size]

    return formatted_chunks


#
# контекст бота
# разделенение текста
# логи в файлик
# сохранять данные


CHROMA_PATH = "chroma"


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Check if chunk is a dictionary and access metadata directly
        if isinstance(chunk, dict):
            source = chunk.get("source")
            page = chunk.get("page")
            if "metadata" not in chunk:
                chunk["metadata"] = {}  # Initialize metadata if not present
        else:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        if isinstance(chunk, dict):
            chunk["metadata"]["id"] = chunk_id
        else:
            chunk.metadata["id"] = chunk_id

    return chunks


CHROMA_PATH = "chroma"


def add_to_chroma(chunks: list):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Assign IDs to chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    across = 0
    new_chunks = []
    for chunk in chunks_with_ids:
        # Access the ID and convert chunk to Document if it’s a dictionary
        chunk_id = chunk.get("metadata", {}).get("id") if isinstance(chunk, dict) else chunk.metadata.get("id")
        if chunk_id not in existing_ids:
            # Convert chunk to Document if it's a dictionary
            if isinstance(chunk, dict):
                chunk = Document(
                    page_content=chunk.get("content", ""),  # Get content or provide a default empty string
                    metadata=chunk.get("metadata", {})
                )
            new_chunks.append(chunk)
        else:
            across += 1

    print(across)

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


# AQVN0A4joVdAjT2FC-8kXP4a8Tmp3bf5buywhC0X


# AQVNwXV4Rl4QNrTw5Kqnbgy7fv3eHyMW6oyit8iw
# aje1els4cngg6qqf1p5a

def get_embedding_function():
    embeddings = YandexGPTEmbeddings(
        folder_id="b1gp04edkv24eujvp7qv",
        api_key='AQVNwXV4Rl4QNrTw5Kqnbgy7fv3eHyMW6oyit8iw')

    return embeddings


from langchain.prompts import ChatPromptTemplate


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    text_prompt = """ 
    Ответьте на вопрос, основываясь только на следующем контексте:
    {context}

    Ответьте на вопрос, основываясь на приведенном выше контексте: 
    {question}
    """

    results = db.similarity_search_with_score(query_text, k=5)
    #
    for i in results:
        print(i)
    print()
    print()
    if len(results) == 0 or results[0][1] > 1.4:
        return "Информация по данному вопросу отсутствует во входных данных."

    context_text = "\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(text_prompt)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)
    promt = {
    "modelUri": "gpt://b1gtm1dktbnprji5rmcp/yandexgpt-lite",
    "completionOptions": {
        "stream": False,
        "temperature": 0.6,
        "maxTokens": 4000
    },
    "messages": [
        {
            "role": "user",
            "text": prompt
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
    data = json.loads(response.text)

    return (data['result']['alternatives'][0]['message']['text'])

#
# documents = load_documents(source_file)
# chunks = split_documents(documents, source_file)
#
# # sorted_chunks = sorted(chunks, key=len, reverse=True)
# #
# # for chunk in sorted_chunks:
# #     print((chunk))
# #     print()
#
#
# add_to_chroma(chunks)
# query_text = "На каком языке осуществляется служебное делопроизводство в РФ?"
# # query_text = "Какой состав Федерального Собрания?"
#
# # input("Введите запрос: ")
# query_rag(query_text)
#
# # На каком языке осуществляется служебное делопроизводство в РФ?
#
#
# # chunks = split_documents(documents)
# # documents = load_documents()
# # print(chunks[0])



