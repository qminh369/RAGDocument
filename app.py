import gradio as gr
import numpy as np
import string
import os
import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from rank_bm25 import BM25Okapi

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load pdf
loader = PDFMinerLoader("/home/skmlab/data/quangminh/RAGFAISS/rule2015.pdf")
text = loader.load()

# chunk
#text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50)
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100)
docs = text_splitter.split_documents(text)

# Initializing the Bi-Encoder model
bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#bi_encoder.max_seq_length = 256  # Truncate long passages to 256 tokens
top_k = 100  # Number of passages to retrieve with the bi-encoder

# List chunks
chunks = []
for doc in docs:
    chunks.append(doc.page_content)

corpus_embeddings = bi_encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

# Cross encoder (reranking)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

# Tokenizing the corpus for BM25
tokenized_corpus = [chunk.split(" ") for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Define a function for search
def search(query):
    # Lexical Search (BM25)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    top_lexical = [chunks[hit['corpus_id']].replace("\n", " ") for hit in bm25_hits[0:5]]
    
    # Semantic Search
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Re-Ranking with Cross-Encoder
    cross_inp = [[query, chunks[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Combine scores and present the results
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

    top_cross_encoder_hits = [chunks[hit['corpus_id']].replace("\n", " ") for hit in hits[0:5]]
    
    result = top_lexical + top_cross_encoder_hits
    return result

# Load model
model_path = "Open-Orca/Mistral-7B-OpenOrca"
#model_path = "SeaLLMs/SeaLLM-7B-Chat"
#model_path = "vilm/vinallama-7b"

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"":0})

def get_relevant(question):
    docs = search(question)
    
    doc_relevant = ""
    for doc in docs:
        doc_relevant += doc + "\n"
    
    return doc_relevant

def write_prompt(doc_relevant, question):
    # Prompt SeaLLM
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #PROMPT = """You are a multilingual, helpful, respectful and honest assistant. Please always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. Your response should adapt to the norms and customs of the respective language and culture.\n\n Bạn sẽ dùng ngữ cảnh được cung cấp để trả lời câu hỏi từ người dùng. Đọc kĩ ngữ cảnh trước khi trả lời câu hỏi và suy nghĩ từng bước một. Dưới đây là ngữ cảnh:\n{doc_relevant}\nHãy trích xuất trong những điều luật đó về nội dung có liên quan đến câu hỏi sau:"{question}"\n Ứng với câu trả lời thu được hãy trích dẫn số hiệu điều luật mà câu trả lời sử dụng, ví dụ "Câu trả lời: ...\nTrích dẫn từ điều: ..."""

    # Prompt Mistral
    PROMPT = """### Instruction:\n\n"Bạn sẽ đóng vai trò là một trợ lý nhằm mục đích trả lời câu hỏi dựa trên tài liệu có liên quan mà người dùng đưa ra. Dưới đây là một vài tài liệu có liên quan đến câu hỏi mà bạn sẽ trả lời:"\n\n{doc_relevant}\n\nHãy đưa ra câu trả lời đầy đủ, ngắn gọn và chính xác nhất. Hãy trích xuất trong những điều luật đó về nội dung có liên quan đến câu hỏi sau:\n\n"{question}"\n\n### Response:"""
    
    input_prompt = PROMPT.format_map(  
    {"doc_relevant": doc_relevant, "question": question}  
)
    return input_prompt

def generate(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt")
    
    outputs = model.generate(  
        inputs=input_ids["input_ids"].to("cuda"),  
        attention_mask=input_ids["attention_mask"].to("cuda"),  
        do_sample=True,  
        max_new_tokens=1024, # 1024  
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  
    )  

    return outputs

def answer(question):
    doc_relevant = get_relevant(question)
    input_prompt = write_prompt(doc_relevant, question)
    outputs = generate(input_prompt)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("### Response:")[1]
    
    return response.strip()

def chatbot(question, history=[]):
  output = answer(question)
  history.append((question, output))
  return history, history

demo = gr.Interface(fn=chatbot,
             inputs=["text", "state"],
             outputs=["chatbot", "state"])

demo.queue().launch(share=True)