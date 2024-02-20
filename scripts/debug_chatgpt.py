import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import openai

# from openai import OpenAI
from openai import AzureOpenAI

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

response = openai_client.chat.completions.create(
    model="chatgpt_icd_coding",
    # response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful clinician's assistant designed to paraphrase clinical statements and to output a YAML list",
        },
        {
            "role": "user",
            "content": "Create 5 full negation statements of: 'All the primary trial participants do not receive any oral capecitabine, oral lapatinib ditosylate or cixutumumab IV, in conrast all the secondary trial subjects receive these'",
        },
    ],
    temperature=0,
    top_p=0,
    max_tokens=1000,
    frequency_penalty=0,
    presence_penalty=0,
)
print(response.choices[0].message.content)
