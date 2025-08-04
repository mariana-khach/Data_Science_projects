# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 22:51:50 2025

@author: Mariana Khachatryan
"""

import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI



load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")


pdf_loader = PyPDFLoader("./pdfs/resume.pdf")
documents=pdf_loader.load()

chain=load_qa_chain(llm=OpenAI(api_key=api_key, verbose=True))
query="What is the email of the person?"
response = chain.invoke({"input_documents":documents,"question":query})

print(f"Output is : {response.get('output_text')}")