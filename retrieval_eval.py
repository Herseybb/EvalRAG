

class RetrievalEvaluator():
    def __init__(self, retrieved_list, gold_list) -> None:
        """_summary_

        Args:
            retrieved_list (List[List[str]]): 
            gold_list (List[List[str]]): 
        """
        assert len(retrieved_list) == len(gold_list), "retrieved_list and gold_list must have same length"
        
        self.retrieved_list = retrieved_list
        self.gold_list = gold_list
    
    def hit_at_k(self, k:int) -> float:
        hits = 0
        for retrieved, gold in zip(self.retrieved_list, self.gold_list):
            topk = retrieved[:k]
            if any(doc in gold for doc in topk):
                hits += 1
        
        return hits / len(self.retrieved_list)

    def mrr(self) -> float:
        rr_total = 0.0
        for retrieved, gold in zip(self.retrieved_list, self.gold_list):
            rr = 0.0
            for rank, doc in enumerate(retrieved, start=1):
                if doc in gold:
                    rr = 1.0 / rank
                    break
            rr_total += rr
        
        return rr_total / len(self.retrieved_list)