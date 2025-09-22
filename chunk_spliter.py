from typing import List
from transformers import AutoTokenizer

# a common transformer tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text_by_token(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    split text into chunks based on token size
    """
    # split based on paragraph
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        token_len = len(tokenizer.encode(para))
        if current_len + token_len <= chunk_size:
            current_chunk.append(para)
            current_len += token_len
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            # new chunk
            current_chunk = [para]
            current_len = token_len
    
    # add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # slip window is need overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            text_chunk = chunks[i]
            if i > 0:
                prev_chunk_tokens = tokenizer.encode(chunks[i-1])
                overlap_tokens = prev_chunk_tokens[-overlap:] if len(prev_chunk_tokens) > overlap else prev_chunk_tokens
                overlap_text = tokenizer.decode(overlap_tokens)
                text_chunk = overlap_text + ' ' + text_chunk
            overlapped_chunks.append(text_chunk)
        chunks = overlapped_chunks
    
    return chunks



if __name__ == '__main__':
    
    # 对每篇文档做 chunk
    chunked_docs = []
    for doc_id, doc in enumerate(docs):
        chunks = chunk_text_by_token(doc['body'], chunk_size=200, overlap=50)
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'doc_id': doc_id,
                'chunk_id': i,
                'title': doc['title'],
                'url': doc['url'],
                'text': chunk
            })

    # 查看前几个 chunk
    for c in chunked_docs[:2]:
        print(c)