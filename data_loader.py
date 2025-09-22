import json

def load_corpus():
    
    with open(r"data/corpus.json", 'r') as f:
        corpus = json.load(f)
    return corpus

def load_qa_pair():
    
    with open(r"data/MultiHopRAG.json", "r") as f:
        qa_pair = json.load(f)
    return qa_pair