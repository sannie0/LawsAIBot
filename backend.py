from langchain_community.document_loaders import PyPDFDirectoryLoader
import requests, json

def load_documents(path):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


from langchain_community.embeddings.yandex import YandexGPTEmbeddings


def get_embedding_function():
    embeddings = YandexGPTEmbeddings(
        folder_id="b1gp04edkv24eujvp7qv",
        api_key='AQVNwXV4Rl4QNrTw5Kqnbgy7fv3eHyMW6oyit8iw')

    return embeddings


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


CHROMA_PATH = "chroma"
from langchain_chroma import Chroma


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("no new documents to add")


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
    print(results)
    if len(results) == 0 or results[0][1] > 1.4:
        return "Информация по данному вопросу отсутствует во входных данных."
    context_text = "\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(text_prompt)
    prompt = prompt_template.format(context=context_text, question=query_text)

    promt = {
        "modelUri": "gpt://b1gtm1dktbnprji5rmcp/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": 2000
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

    return(data['result']['alternatives'][0]['message']['text'])