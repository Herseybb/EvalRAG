import faiss
import numpy as np



def embed(chunks, model):
    texts = [doc["text"] for doc in chunks]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu().numpy()
    
    return embeddings

def faiss_index(chunks, model, save):
    
    embeddings = embed(chunks, model)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    index.add(embeddings)
        
    return index

