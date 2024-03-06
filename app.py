from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json
import re
import torch
from torch import cuda


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

biomistral_local_llm = "./biomistral-quantized-version/biomistral-finetuned-7b-v2.1-gguf-unsloth.Q4_K_M.gguf"
alphamonarch_local_llm = "./alpha-monarch-7B-quantized-version/alpha-monarch-7B-gguf-unsloth.Q4_K_M.gguf"
gemma_local_llm = "./gemma-7B-quantized-version/content/gemma-7b-v4.0-fine-tuned.gguf"


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)


from accelerate import Accelerator

if cuda.is_available():
    accelerator = Accelerator()
    gpu_layers = 50

else:
    gpu_layers = 0

print(gpu_layers)

# ./alpha-monarch-7B-quantized-version/alpha-monarch-7B-gguf-unsloth.Q4_K_M.gguf

# Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path= biomistral_local_llm,
#     n_ctx=32768,
#     n_threads=8,
#     n_gpu_layers=gpu_layers,
#     temperature=0.3,
#     max_tokens=2048,
#     top_p=1,
#     grammar_path="./grammar.gbnf",
# )

llm = LlamaCpp(
    model_path=alphamonarch_local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)

print("LLM Initialized....")

prompt_template = """
Below is a question posed by healthcare professionals
including nurses, doctors, and researchers in Malawi,
all of whom actively engage in disease surveillance efforts.

Context: {context}

### Question:
{question}

Offer a response that is both accurate and concise, incorporating
relevant keywords to address the inquiry effectively.
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})


def regex_to_preprocess_text_from_biomistral(text):
    answer_text = ""
    keywords = []

    # Define patterns for extracting information
    response_pattern = r"### Answer:\n(.*?)\n\n"
    keywords_pattern = r"### Keywords:\n(.*?)\n\n"

    # Extract information using regular expressions
    response_match = re.search(response_pattern, text)
    keywords_match = re.search(keywords_pattern, text)

    # Check if matches were found and extract information
    if response_match:
        answer_text = response_match.group(1)

    if keywords_match:
        keywords = keywords_match.group(1).strip()

    # Return the extracted information
    return answer_text, keywords


def regex_to_obtain_paragraphs_number(source_document):
    paragraphs_number = None

    # Define the regular expression pattern
    paragraphs_number_pattern = r'(\d+)\n'

    # Search for the pattern in the source document
    paragraphs_number_match = re.search(paragraphs_number_pattern, source_document)

    # Extract the paragraph number or page number if the pattern was found
    if paragraphs_number_match:
        paragraphs_number = paragraphs_number_match.group(1)

    # Return the extracted paragraph number
    return paragraphs_number


def regex_to_obtain_file_name_and_get_booklet(doc):
    # Define the regular expression pattern
    pattern = r'([^/]+)$'

    # Search for the pattern in the text
    matches = re.search(pattern, doc)

    # Extract the file name if found
    if matches:
        file_name = matches.group(1)

    file_booklet_name = {
        "TG_Booklet_1.xlsx": "TG Booklet 1",
        "TG_Booklet_2.xlsx": "TG Booklet 2",
        "TG_Booklet_3.xlsx": "TG Booklet 3",
        "TG_Booklet_4.xlsx": "TG Booklet 4",
        "TG_Booklet_5.xlsx": "TG Booklet 5",
        "TG_Booklet_6.xlsx": "TG Booklet 6"
    }

    # Return the correspondent booklet name based on the file name
    return file_booklet_name[file_name]



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    print(response)
    answer = response['result']
    print(answer)
    answer_, keywords = regex_to_preprocess_text_from_biomistral(answer)
    print(answer_)
    print(keywords)
    source_document = response['source_documents'][0].page_content
    print(source_document)
    paragraphs_number = regex_to_obtain_paragraphs_number(source_document)
    print(paragraphs_number)
    doc = response['source_documents'][0].metadata['source']
    print(doc)
    file_booklet_name = regex_to_obtain_file_name_and_get_booklet(doc)
    print(file_booklet_name)
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    print(response_data)
    res = Response(response_data)
    return res
