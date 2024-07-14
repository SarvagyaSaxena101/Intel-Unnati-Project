import os 
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import huggingface
from langchain_community.vectorstores import faiss
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st 

model = GoogleGenerativeAI(google_api_key='AIzaSyDQViky6jXUHMHNSlD_tcTUQ06QYa2Ya_A',temperature=0.8,model='models/text-bison-001')

df = CSVLoader('indian_food.csv',source_column='name')
data = [row for row in df.lazy_load() if row is not None]

instructor_embedding = huggingface.HuggingFaceEmbeddings()


vector_db = faiss.FAISS.from_documents(documents=data,embedding=instructor_embedding)

retriever_db = vector_db.as_retriever()


prompt_template = '''Given the following context and a question, generate a answer based on this context.In the answer try to provide as much text as possible from all the
sections in the source document. If the answer is not found in the context kindly say that u dont know the answer, Dont try to make up a answer
CONTEXT:{context}
QUESTION:{question}'''


PROMPT = PromptTemplate(template=prompt_template,input_variables=['context','question' ])

chain = RetrievalQA.from_chain_type(llm = model,chain_type ="stuff",retriever=retriever_db,input_key = "query",return_source_documents = False,chain_type_kwargs={"prompt":PROMPT})
recipe_question = input("Enter the dish name: ")
response = chain(f"Give me the recipe of {recipe_question}")


sentence = ''
for words in response['result']:
    if words != '\n':
        sentence += words
    else:
        print(sentence)
        sentence = ''

