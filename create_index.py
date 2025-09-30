from data_loader import load_corpus
from chunk_spliter import chunk_text_by_token
from indexer import faiss_index
import faiss

import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(r"multi-qa-mpnet-base-dot-v1")

def create_index():
    corpus = load_corpus()
    
    # split chunks
    chunked_docs = []
    for doc_id, doc in enumerate(corpus):
        chunks = chunk_text_by_token(doc['body'], chunk_size=300, overlap=50)
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'doc_id': doc_id,
                'chunk_id': i,
                'title': doc['title'],
                'author': doc['author'],
                'source': doc['source'],
                'published_at': doc['published_at'],
                'category': doc['category'],
                'url': doc['url'],
                'text': chunk
            })

    # save chunks
    with open("data/metadata3.pkl", "wb") as f:
        pickle.dump(chunked_docs, f)
    
    # index
    index = faiss_index(chunked_docs, model)
    # save index
    faiss.write_index(index, "data/index3.index")


if __name__ == '__main__':
    create_index()