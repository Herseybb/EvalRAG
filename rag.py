from data_loader import load_corpus, load_qa_pair
from chunk_spliter import chunk_text_by_token
from indexer import faiss_index
from llm.llm_farm import get_response
from retrieval_eval import RetrievalEvaluator
from generation_eval import GenerationEvaluator


import pickle
import faiss
import numpy as np
import pandas as pd
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
    retrieved_chunks = [chunks[i] for i in indices[0]]
    retrieved_texts = [chunks[i]["text"] for i in indices[0]]    
    
    # create prompt
    context = "\n\n".join(retrieved_texts)
    prompt = ""\
        "Use the following context to answer the question as accurately as possible."\
        "\n"\
        "Context:\n"\
        f"{context}"\
        '\n'\
        f"Question: {query}"\
        "\n"\
        "Answer:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
        "content": prompt}
                ]
    response = get_response(messages=messages, stream=False)
    
    return response, retrieved_chunks

if __name__ == "__main__":
    
    # load chunks metadata and index
    chunks, index = load_index()
    
    qa_pair = load_qa_pair()
    
    qa_pair = qa_pair[:10] # for test
    
    # query loop
    retrieved_ids = []
    gold_ids = []
    gold_answers = []
    generation_answers = []
    
    results = []
    for qa in qa_pair:
        query = qa['query']
        answer, retrieved_chunks = generate_answer(chunks, query, top_k=5)

        # retrieval metrics
        # retrieved doc ids (in lists)
        retrieved_id = [chunk['doc_id']  for chunk in retrieved_chunks]
        
        # gold doc ids
        gold_id = []
        for evidence in qa['evidence_list']:
            doc_id = next(
                (d['doc_id'] for d in chunks
                 if d["title"]==evidence['title'] and d['published_at']==evidence['published_at']),
                None
            )
            gold_id.append(doc_id)
        
        
        # metrics
        retrieval_evaluator = RetrievalEvaluator(retrieved_list=[retrieved_id], gold_list=[gold_id])
        hit_at_1 = retrieval_evaluator.hit_at_k(1)
        hit_at_3 = retrieval_evaluator.hit_at_k(3)
        hit_at_5 = retrieval_evaluator.hit_at_k(5)
        rr = retrieval_evaluator.mrr()
        
        gen_evaluator = GenerationEvaluator(generation_list=[answer], gold_list=[qa['answer']])
        score_meteor = gen_evaluator.meteor()
        
        retrieved_ids.append(retrieved_id)
        gold_ids.append(gold_id)
        gold_answers.append(qa['answer'])
        generation_answers.append(answer)
        
        # save results
        results.append({
            "query": query,
            "gen_answer": answer,
            "gold_answer": qa['answer'],
            "gold_ids": gold_id,
            "retrieved_ids": retrieved_id,
            "hit@1": hit_at_1,
            "hit@3": hit_at_3,
            "hit@5": hit_at_5,
            "rr": rr,
            "meteor": score_meteor
        })
        
    # metrics
    retrieval_evaluator = RetrievalEvaluator(retrieved_list=retrieved_ids, gold_list=gold_ids)
    gen_evaluator = GenerationEvaluator(generation_list=generation_answers, gold_list=gold_answers)
    
    hit_at_1 = retrieval_evaluator.hit_at_k(1)
    hit_at_3 = retrieval_evaluator.hit_at_k(3)
    hit_at_5 = retrieval_evaluator.hit_at_k(5)
    meteor = gen_evaluator.meteor()
    
    df_result = pd.DataFrame(results)
    df_result.to_csv(r"result/result.csv", index=None)
    
    
    
        
    
        
        
    print(query)
    print(answer)