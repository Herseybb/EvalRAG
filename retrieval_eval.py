

class RetrievalEvaluator:
    def __init__(self, retrieved_list, gold_list) -> None:
        """
        Args:
            retrieved_list (List[List[str]]): 
            gold_list (List[List[str]]): 
        """
        assert len(retrieved_list) == len(gold_list), \
            "retrieved_list and gold_list must have same length"
        
        self.retrieved_list = retrieved_list
        self.gold_list = gold_list

    def hit_at_k(self, k: int):
        hits = 0
        valid_queries = 0  # only count queries with non-empty gold

        for retrieved, gold in zip(self.retrieved_list, self.gold_list):
            if not gold:  # skip empty gold list
                continue
            valid_queries += 1
            topk = retrieved[:k]
            if any(doc in gold for doc in topk):
                hits += 1

        if valid_queries == 0:
            return None
        return hits / valid_queries

    def mrr(self):
        rr_total = 0.0
        valid_queries = 0  # only count queries with non-empty gold

        for retrieved, gold in zip(self.retrieved_list, self.gold_list):
            if not gold:  # skip empty gold list
                continue
            valid_queries += 1
            rr = 0.0
            for rank, doc in enumerate(retrieved, start=1):
                if doc in gold:
                    rr = 1.0 / rank
                    break
            rr_total += rr

        if valid_queries == 0:
            return None
        return rr_total / valid_queries
    
    def precision_at_k(self, k: int) -> float:
        total_precision = 0.0
        valid_queries = 0

        for retrieved, gold in zip(self.retrieved_list, self.gold_list):
            topk = retrieved[:k]

            # when gold list is empty
            if not gold:
                if not topk:
                    # gold empty + retrieve empty → precision = 1
                    total_precision += 1.0
                else:
                    # gold empty + retrieve something → precision = 0
                    total_precision += 0.0
                valid_queries += 1
                continue

            # gold list is not empty
            if not topk:
                total_precision += 0.0
            else:
                num_relevant = sum(1 for doc in topk if doc in gold)
                total_precision += num_relevant / len(topk)

            valid_queries += 1

        if valid_queries == 0:
            return 0.0
        return total_precision / valid_queries

