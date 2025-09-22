import os
from openai import OpenAI
import re
from dotenv import load_dotenv

# setting proxy
import sys, os, os.path
import truststore
import time

load_dotenv()

class llmfarminf():
    def __init__(self, model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B") -> None:
        truststore.inject_into_ssl()


        self.client = OpenAI(
            default_headers={"KeyId": os.environ["OPENAI_API_KEY"]}
        )
        self.model = model

    def _completion_stream(self, messages, response_format=None):
        
        stream_resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_completion_tokens=50000,
            response_format=response_format
        )
        for chunk in stream_resp:
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta and hasattr(delta, "content") and delta.content:
                yield delta.content
    def _completion(self, messages, response_format=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=50000,
            response_format=response_format
        )
        content = response.choices[0].message.content
        # cut think part
        content = content.split("</think>", 1)[-1]
        return content

def get_response(messages, response_format=None, stream=False):
    obj = llmfarminf()

    if stream:
        return obj._completion_stream(messages, response_format)
    else:
        return obj._completion(messages, response_format)

