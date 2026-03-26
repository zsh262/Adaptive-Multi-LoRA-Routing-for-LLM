import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from router.domain_config import DOMAINS

class EmbeddingRouter:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        model_name = os.getenv("ROUTER_EMBEDDING_MODEL", model_name)
        print(f"Loading Router Embedding Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # 预计算所有领域的 Embedding
        self.domain_names = list(DOMAINS.keys())
        domain_texts = list(DOMAINS.values())
        self.domain_embeddings = self._get_embeddings(domain_texts)
        print("Router initialized successfully.")

    def _get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        计算输入文本的 Embedding 向量。
        """
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def route(self, query: str) -> tuple[str, dict[str, float]]:
        """
        接收用户 Query，返回最匹配的领域以及所有领域的得分。
        """
        # 1. 计算 Query 的 Embedding
        query_embedding = self._get_embeddings([query])
        
        # 2. 计算余弦相似度 (Cosine Similarity)
        # query_embedding: [1, hidden_size]
        # domain_embeddings: [num_domains, hidden_size]
        # 结果 scores: [1, num_domains]
        scores = torch.mm(query_embedding, self.domain_embeddings.transpose(0, 1))[0]
        
        # 3. 整理得分并找出最高分
        domain_scores = {self.domain_names[i]: float(scores[i]) for i in range(len(self.domain_names))}
        best_domain = max(domain_scores, key=domain_scores.get)
        
        return best_domain, domain_scores

# 简单测试代码
if __name__ == "__main__":
    router = EmbeddingRouter()
    
    test_queries = [
        "Write a Python function to reverse a string.",
        "Can you summarize the main contributions of the Attention is All You Need paper?",
        "How does the VITS model achieve high-quality text-to-speech synthesis?"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        best_domain, scores = router.route(q)
        print(f"Selected Domain: {best_domain}")
        print(f"Scores: {scores}")
