import numpy as np

class Ranker:
    def __init__(self, behaviors, history, articles):
        """
        Initialize the CombinedAttentionPromptModel with a NAML model, prompts, and a weight for combining scores.
        
        :param naml_model: Trained NAML model
        :param prompts: List of prompt strings
        :param weight: Weight for combining original and prompted attention scores (default is 0.5)
        """
        # self.users = users
        # self.impressions = impressions
        self.articles = articles
   
    

