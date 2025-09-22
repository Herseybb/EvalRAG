from openai import OpenAI
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class llmfarminf():
    def __init__(self, model = "gpt-4o-mini") -> None:
        self.client = OpenAI(
            api_key="dummy",
            base_url=os.environ['base_url'],
            default_headers = {"genaiplatform-farm-subscription-key": os.environ['genaiplatform-farm-subscription-key']}
        )
        self.model = model

    def _completion_stream(self, messages, response_format=None):
        # stram response
        stream_resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            extra_query={"api-version": "2024-08-01-preview"},
            response_format=response_format
        )
        for chunk in stream_resp:
            if not chunk.choices:
                continue
            
            # 安全检查每个 chunk 是否有 delta 和内容
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta and hasattr(delta, "content") and delta.content:
                yield delta.content
    
    def _completion(self, messages, response_format=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_query={"api-version": "2024-08-01-preview"},
            response_format=response_format
        )
        return response.choices[0].message.content

def get_response(messages, response_format=None, stream=False):
    import os
    
    obj = llmfarminf()

    if stream:
        return obj._completion_stream(messages, response_format)
    else:
        return obj._completion(messages, response_format)