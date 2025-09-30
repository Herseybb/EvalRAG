from data_loader import load_qa_pair
from llm.llm_farm import get_response
from retrieval_eval import RetrievalEvaluator
from generation_eval import GenerationEvaluator


import pickle
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer(r"C:\Users\cui8szh\.cache\huggingface\hub\models--sentence-transformers--multi-qa-mpnet-base-dot-v1\snapshots\\17997f24dca0df1a4fed68894fb0e1e133e60482")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_index():
    index = faiss.read_index("data/index3.index")
    with open("data/metadata3.pkl", 'rb') as f:
        chunks = pickle.load(f)
        
    return chunks, index


def retrieve(index, query, topk=50):
    # query embedding
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    
    # retrival topk
    distances, indices = index.search(q_vec, topk)

    return indices[0]


def rerank(indices, query, chunks, top_k=5):
    
    retrieved_texts = [chunks[i]["text"] for i in indices]
    
    # reranker
    scores = reranker.predict([(query, text) for text in retrieved_texts])
    indices, scores = zip(*sorted(zip(indices, scores), key=lambda x: x[1], reverse=True))
    
    indices = indices[ : top_k]
    
    return indices



def generate_answer(retrieved_texts, query):
        
    # create prompt
    context = "\n\n".join(retrieved_texts)
    prompt = ""\
        "Use and ONLY use the following context to answer the question as accurately as possible.\n"\
        "Keep your answer concise (use words and phrases instead of sentences)\n"\
        "\n\n"\
        "Context:\n"\
        f"{context}"\
        '\n\n'\
        f"Question: {query}"\
        "\n\n"\
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
    
    # qa_pair = qa_pair[20:30] # for test
    
    # query loop
    retrieved_ids = []
    gold_ids = []
    gold_answers = []
    generation_answers = []
    
    results = []
    rerank_eval = []  # for evaluation the effect of reranking
    for qa in tqdm(qa_pair):
        query = qa['query']
        
        # retrieval
        indices_retrieved = retrieve(index, query, topk=50)
        
        # re-rank
        indices_reranked = rerank(indices_retrieved, query, chunks, top_k=5)
        
        # retrieved chunks and texts
        retrieved_texts = [chunks[i]["text"] for i in indices_reranked]
        
        # generate answers
        # answer = generate_answer(retrieved_texts, query)
        
        # gold doc ids
        gold_id = []
        for evidence in qa['evidence_list']:
            doc_id = next(
                (d['doc_id'] for d in chunks
                 if d["title"]==evidence['title'] and d['published_at']==evidence['published_at']),
                None
            )
            gold_id.append(doc_id)
        
        
        # retrieval metrics
        retrieved_chunks = [chunks[i] for i in indices_reranked]
        retrieved_id = [chunk['doc_id']  for chunk in retrieved_chunks]
        retrieval_evaluator = RetrievalEvaluator(retrieved_list=[retrieved_id], gold_list=[gold_id])
        hit_at_1 = retrieval_evaluator.hit_at_k(1)
        hit_at_3 = retrieval_evaluator.hit_at_k(3)
        hit_at_5 = retrieval_evaluator.hit_at_k(5)
        rr = retrieval_evaluator.mrr()
        precision_at_1 = retrieval_evaluator.precision_at_k(1)
        precision_at_3 = retrieval_evaluator.precision_at_k(3)
        precision_at_5 = retrieval_evaluator.precision_at_k(5)
        
        
        retrieved_chunks = [chunks[i] for i in indices_retrieved]
        retrieved_id_retrieve = [chunk['doc_id']  for chunk in retrieved_chunks]
        retrieval_evaluator = RetrievalEvaluator(retrieved_list=[retrieved_id_retrieve], gold_list=[gold_id])
        hit_at_1_retrieve = retrieval_evaluator.hit_at_k(1)
        hit_at_3_retrieve = retrieval_evaluator.hit_at_k(3)
        hit_at_5_retrieve = retrieval_evaluator.hit_at_k(5)
        rr_retrieve = retrieval_evaluator.mrr()
        precision_at_1_retrieve = retrieval_evaluator.precision_at_k(1)
        precision_at_3_retrieve = retrieval_evaluator.precision_at_k(3)
        precision_at_5_retrieve = retrieval_evaluator.precision_at_k(5)
        
        rerank_eval.append({
            "query": query,
            "gold_ids": gold_id,
            "retrieve retrieved_ids": retrieved_id_retrieve,
            "retrieve hit@1": hit_at_1_retrieve,
            "retrieve hit@3": hit_at_3_retrieve,
            "retrieve hit@5": hit_at_5_retrieve,
            "retrieve rr": rr_retrieve,
            "retrieve precision@1": precision_at_1_retrieve,
            "retrieve precision@3": precision_at_3_retrieve,
            "retrieve precision@5": precision_at_5_retrieve,
            
            "rerank retrieved_ids": retrieved_id,
            "rerank hit@1": hit_at_1,
            "rerank hit@3": hit_at_3,
            "rerank hit@5": hit_at_5,
            "rerank rr": rr,
            "rerank precision@1": precision_at_1,
            "rerank precision@3": precision_at_3,
            "rerank precision@5": precision_at_5,
            
        })
        
        
        
        # generation metrics
        # gen_evaluator = GenerationEvaluator(generation_list=[answer], gold_list=[qa['answer']])
        # score_meteor = gen_evaluator.meteor()
        # score_embed = gen_evaluator.embed(model=model)
        
        
    #     retrieved_ids.append(retrieved_id)
    #     gold_ids.append(gold_id)
    #     gold_answers.append(qa['answer'])
    #     generation_answers.append(answer)
        
    #     # save results
    #     results.append({
    #         "query": query,
    #         "gen_answer": answer,
    #         "gold_answer": qa['answer'],
    #         "gold_ids": gold_id,
    #         "retrieved_ids": retrieved_id,
    #         "hit@1": hit_at_1,
    #         "hit@3": hit_at_3,
    #         "hit@5": hit_at_5,
    #         "rr": rr,
    #         "meteor": score_meteor,
    #         "embed": score_embed
    #     })
        
    # # metrics
    # retrieval_evaluator = RetrievalEvaluator(retrieved_list=retrieved_ids, gold_list=gold_ids)
    # gen_evaluator = GenerationEvaluator(generation_list=generation_answers, gold_list=gold_answers)
    
    # hit_at_1 = retrieval_evaluator.hit_at_k(1)
    # hit_at_3 = retrieval_evaluator.hit_at_k(3)
    # hit_at_5 = retrieval_evaluator.hit_at_k(5)
    # meteor = gen_evaluator.meteor()
    
    # df_result = pd.DataFrame(results)
    # df_result.to_csv(r"result/result.csv", index=None)
    
    df_rerank_eval = pd.DataFrame(rerank_eval)
    df_rerank_eval.to_excel(r"result/result_index3.xlsx", index=None)