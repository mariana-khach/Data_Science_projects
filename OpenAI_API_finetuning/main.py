# -*- coding: utf-8 -*-
"""
Spyder Editor

author: Mariana Khachatryan 7/31/2025
"""

import json
from openai import OpenAI
from fastapi import FastAPI
import os

from dotenv import load_dotenv
load_dotenv()

app=FastAPI()
#Getting OPENAI API key
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(f"OpenAI API key is = {os.getenv('OPENAI_API_KEY')}")

@app.post("/v1/chat/completions")
def chat_completions(usr_prompt:str):
    system_prompt="You are a text classification model, given an input text, \
        you will return a JSON object containing the probability scores for\
            the following categories: 'label'. The JASON object should have\
                key corresponding to these category, and the values should be\
        0 if email is classified as spam and 1 if email is not a spam. \
            Please respond with only the JSON object, without any additional \
                text or explanation."
                
    model="ft:gpt-3.5-turbo-0125:personal::BzD1XHjE"
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"system_prompt"},
            {"role":"user","content":"user_prompt"},
            ]
        
        )

    return json.loads(response.choices[0].message.content)