from data_loader import load_corpus, load_qa_pair
from chunk_spliter import chunk_text_by_token
from indexer import faiss_index
from llm.llm_farm import get_response


import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_index():
    corpus = load_corpus()
    
    # split chunks
    chunked_docs = []
    for doc_id, doc in enumerate(corpus):
        chunks = chunk_text_by_token(doc['body'], chunk_size=200, overlap=50)
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
    with open("data/metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    # index
    index = faiss_index(chunked_docs, model)
    # save index
    faiss.write_index(index, "data/index1.index")

def load_index():
    index = faiss.read_index("data/index1.index")
    with open("data/metadata.pkl", 'rb') as f:
        chunks = pickle.load(f)
        
    return chunks, index


def generate_answer(chunks, query, top_k=3):
    # query embedding
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    
    # retrival topk
    distances, indices = index.search(q_vec, top_k)
    retrieved_chunks = [chunks[i]["text"] for i in indices[0]]    
    
    # create prompt
    context = "\n\n".join(retrieved_chunks)
    prompt = ""\
        "Use the following context to answer the question as accurately as possible."\
        "\n"\
        "Context:\n"\
        f"{context}"\
        '\n'\
        "Question: {query}"\
        "\n"\
        "Answer:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
        "content": prompt}
                ]
    response = get_response(messages=messages, stream=False)
    
    return response

if __name__ == "__main__":
    
    # load chunks metadata and index
    chunks, index = load_index()
    
    # query example
    qa_pair = load_qa_pair()
    query = qa_pair[0]['query']
    answer = generate_answer(chunks, query)
    # query_vec = model.encode([query], convert_to_numpy=True)

    # k = 2  # top-k
    # distances, indices = index.search(query_vec, k)
    # print("\n🔎 Query:", query)
    # print("Results:")
    # for i, idx in enumerate(indices[0]):
    #     doc = chunks[idx]
    #     print(f"Rank {i+1} | Score {distances[0][i]:.4f}")
    #     print(f"Title: {doc['title']}")
    #     print(f"URL: {doc['url']}")
    #     print(f"Text: {doc['text'][:200]}...\n")
    
    # print(qa_pair[0]['evidence_list']) 
        
        
    print(query)
    print(answer)