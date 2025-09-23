from nltk.translate.meteor_score import meteor_score

class GenerationEvaluator:
    def __init__(self, generation_list, gold_list):
        """
        :param gold_list: list of reference answers
        :param generation_list: list of generated answers 
        """
        assert len(gold_list) == len(generation_list), "gold_list and generation_list must have same length"
        self.gold_list = gold_list
        self.generation_list = generation_list

    def meteor(self):
        """
        """
        scores = []
        for gen, gold in zip(self.generation_list, self.gold_list):
            score = meteor_score([gold.split()], gen.split())
            scores.append(score)
        return sum(scores) / len(scores)
