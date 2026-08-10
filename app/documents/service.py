from nomic import embed
import io
from pypdf import PdfReader
import pandas as pd
from PIL import Image
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from app.core.config import settings
from app.documents.clients import chroma_client,groq_client
import uuid
import base64

def extract_text_plain(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    text=stream.read().decode('utf-8',errors='ignore')
    return text,{'char_count':len(text)}


def extract_text_pdf(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    stream.seek(0)
    reader=PdfReader(stream)
    pages_text=[]
    for i,page in enumerate(reader.pages):
        page_text=page.extract_text()
        if page_text:
            pages_text.append(f'[page {i+1}]\n{page_text}')
    full_text="\n".join(pages_text)
    return full_text,{'total_pages':len(reader.pages)}


def extract_text_html(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    stream.seek(0)
    content=stream.read().decode('utf-8',errors='ignore')
    soup=BeautifulSoup(content,'html.parser')
    text=soup.get_text(separator='\n',strip=True)
    title=soup.title.string if soup.title else ""
    return text,{'title':title}


def extract_text_markdown(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    stream.seek(0)
    md_content=stream.read().decode('utf-8',errors='ignore')
    return md_content,{}

def tabular(df):
    row_str=[]
    i=0
    for _,row in df.iterrows():
        row=" | ".join(f'{col}:{val}' for col,val in row.items())
        row_str.append(row)
        i+=1
    clean_text="\n".join(row_str)
    return clean_text,i
def extract_text_csv(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    stream.seek(0)
    df=pd.read_csv(stream)
    clean_text,i=tabular(df)
    return clean_text,{'num_of_rows':i}


def extract_text_excel(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    stream.seek(0)
    df=pd.read_excel(stream)
    clean_text,i=tabular(df)
    return clean_text,{'num_of_rows':i}


def extract_text_image(stream: io.BytesIO, filename: str)->tuple[str,dict]:
    """
    Extracts text and layout details using Groq's low-cost Llama 3.2 Vision model.
    Runs entirely in memory to prevent RAM overhead on Render backends.
    """
    stream.seek(0)
    try:
        with Image.open(stream) as img:
            metadata = {"img_height": img.height,"img_width":img.width, "format": img.format}
            img_format = img.format.lower() if img.format else "png"
    except Exception as e:
        return f"Error opening image file: {str(e)}", {}

    stream.seek(0)
    file_bytes = stream.read()
    base64_image = base64.b64encode(file_bytes).decode('utf-8')
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is an image that may contain diagrams, flowcharts, or text blocks. "
                                "Transcribe all text, format data structures into markdown tables, and "
                                "explicitly describe any visual flows or shapes in detailed text paragraphs."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1
        )
        extracted_markdown = response.choices[0].message.content
        return str(extracted_markdown), metadata
    except Exception as e:
        fallback = f'ocr failed: {str(e)}'
        return fallback, metadata




def ingestion_pipeline(file_contents:bytes,filename:str):
    file_stream=io.BytesIO(file_contents)
    extension=Path(filename).suffix.lower()

    base_metadata={
        'source':filename,
        'file_type': extension,
        'file_size_kb':round(len(file_contents)/1024)
    }
    parser_dict={
        ".txt":extract_text_plain,
        ".js":extract_text_plain,
        ".py":extract_text_plain,
        ".log":extract_text_plain,
        ".json":extract_text_plain,
        ".pdf":extract_text_pdf,
        ".html":extract_text_html,
        ".htm":extract_text_html,
        ".md":extract_text_markdown,
        ".markdown":extract_text_markdown,
        ".csv":extract_text_csv,
        ".xlsx":extract_text_excel,
        ".xls":extract_text_excel,
        ".jpg":extract_text_image,
        ".jpeg":extract_text_image,
        ".png":extract_text_image,
        ".webp":extract_text_image,
    }
    try:
        if extension in parser_dict:
            text,custom_meta=parser_dict[extension](file_stream,filename)
            base_metadata.update(custom_meta)
            return text,base_metadata
        else:
            raise ValueError(f'Extension {extension} is not allowed')
        
    except Exception as e:
        raise Exception(f'error failed to ingest file{str(e)}')

splitter=RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=['\n\n','\n','. ' # main separators
    ,' ',"" #fallback if exceed size
    ]
)
def chunking(text:str,metadata):
    chunks=splitter.split_text(text)
    docs=[]
    for i ,chunk in enumerate(chunks):
        meta=metadata.copy()
        meta.update({
                'chunk_id':i,
                'total_chunks':len(chunks)
            })
        docs.append({
            'text':chunk,
            'metadata':meta
            }
            )
    return docs


# Creates a persistent database folder in your project directory root


def embed_n_store(chunk_list:list[dict],user_id: uuid.UUID,chat_id:uuid.UUID,chroma_client):

    collection_name = f"chat_{user_id.hex[:12]}"
    
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    documents = []
    metadatas = []
    ids = []
    
    for idx,chunk in enumerate(chunk_list):
        chunk_text=chunk.get('text',"").strip()
        meta=chunk.get('metadata',{}).copy()
        meta.update({
            'user_id':str(user_id),
            "chat_title":str(chat_id)
        })
        documents.append(chunk_text)
        metadatas.append(meta)
        source=meta.get('source','unkown')
        ids.append(f'user_{str(user_id)}_chat_{chat_id}_source_{source}_id_{idx}')

    if not documents:
        return "No text to embed."

    # Fetch vectors from Nomic cloud API
    batch_size=32
    all_embeddings=[]
    for i in range(0,len(documents),batch_size):
        nomic_response =embed.text(
            texts=documents[i:i+batch_size],
            model="nomic-embed-text-v1.5",
            task_type="search_document"
        )
        all_embeddings.extend(nomic_response['embeddings'])
        
        # Store everything in Chroma DB
    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    return collection_name



def full_pipeline(file_contents:bytes,filename:str,uid: uuid.UUID,chat_id:uuid.UUID):
    if file_contents!=None:
        text,metadata=ingestion_pipeline(file_contents,filename)
        chunked_docs=chunking(text=text,metadata=metadata)
        collection_name=embed_n_store(chunk_list=chunked_docs,user_id=uid,chat_id=chat_id,chroma_client=chroma_client)
    return {'no_of_chunk':len(chunked_docs),'metadata':metadata,'collection_name':collection_name}   



async def retrieve_chunks(query:str,user_id:uuid.UUID,chroma_client,top_k:int=5)->list[dict]:
    collection_name = f"chat_{user_id.hex[:12]}"
    try:
        collection=chroma_client.get_collection(name=collection_name)
    except Exception:
        return []
    nomic_response =embed.text(
                texts=[query],
                model="nomic-embed-text-v1.5",
                task_type="search_query"
            )
    query_embedding=nomic_response['embeddings'][0]
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks_retrieved=[]
    if results['documents']:
        for i in range(len(results['documents'][0])):
            chunks_retrieved.append({
                'text':results['documents'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else None
            })
    return chunks_retrieved
async def multi_query_retrieval(query_list: list[str], user_id: uuid.UUID, chroma_client,top_k: int = 5):
   
    tasks=[
        retrieve_chunks(q,user_id,chroma_client,top_k)
        for q in query_list
    ]
    result=await asyncio.gather(*tasks)
    res=[{'query':q,'chunks':chunks}
        for q,chunks in zip(query_list,result)]
    return res

def dedup(retrievals:list[dict]):
    unique_chunks = {}

    for retrieval in retrievals:
        for chunk in retrieval["chunks"]:
            unique_chunks[chunk["id"]] = chunk

    context = list(unique_chunks.values())
    return context