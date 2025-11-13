# from typing import List
# import joblib
# import numpy as np
# from config import settings

# class PCAEmbeddings:
#     """PCA 기반 임베딩 (512차원)"""
    
#     def __init__(self):
#         print(f"PCA 모델 로딩 중: {settings.PCA_MODEL_PATH}")
        
#         # PCA 모델 로드
#         self.pca = joblib.load(settings.PCA_MODEL_PATH)
#         print(f"✅ PCA 모델 로드 완료: {self.pca.n_components_}차원")
        
#         # GPU 여부에 따라 분기
#         if settings.USE_GPU:
#             print("⚠️  GPU 모드: Mock 임베딩 사용 (개발용)")
#             self.use_mock = True
#         else:
#             print("CPU 모드: OpenAI 임베딩 사용")
#             from openai import OpenAI
#             self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
#             self.use_mock = False
        
#         print("✅ 임베딩 준비 완료!")
    
#     def _get_mock_embedding(self, text: str) -> np.ndarray:
#         """Mock 임베딩 생성 (GPU 환경용)"""
#         # 텍스트를 시드로 사용해서 일관성 유지
#         seed = hash(text) % (2**32)
#         np.random.seed(seed)
        
#         # PCA 입력 차원에 맞는 랜덤 벡터
#         mock_embedding = np.random.randn(self.pca.n_features_in_)
        
#         # 정규화 (실제 임베딩처럼 보이게)
#         mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        
#         return mock_embedding
    
#     def _get_openai_embedding(self, text: str) -> np.ndarray:
#         """OpenAI 임베딩 생성 (CPU 환경용)"""
#         response = self.client.embeddings.create(
#             model="text-embedding-3-large",
#             input=text
#         )
        
#         embeddings = np.array(response.data[0].embedding)
        
#         # 차원 맞추기
#         if len(embeddings) != self.pca.n_features_in_:
#             if len(embeddings) > self.pca.n_features_in_:
#                 embeddings = embeddings[:self.pca.n_features_in_]
#             else:
#                 embeddings = np.pad(
#                     embeddings, 
#                     (0, self.pca.n_features_in_ - len(embeddings))
#                 )
        
#         return embeddings
    
#     def embed_query(self, text: str) -> List[float]:
#         """단일 쿼리 임베딩"""
#         # GPU/CPU에 따라 다른 임베딩 사용
#         if self.use_mock:
#             embeddings = self._get_mock_embedding(text)
#         else:
#             embeddings = self._get_openai_embedding(text)
        
#         # PCA 변환
#         pca_embedding = self.pca.transform([embeddings])[0]
        
#         return pca_embedding.tolist()
    
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """여러 문서 임베딩"""
#         return [self.embed_query(text) for text in texts]

# # 싱글톤 인스턴스
# pca_embeddings = PCAEmbeddings()
from typing import List
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from config import settings

class PCAEmbeddings:
    """
    Alibaba GTE 7B + PCA 임베딩 (512차원)
    - 벡터 DB 생성 시와 동일한 파이프라인
    """
    
    def __init__(self):
        print("="*70)
        print("🚀 PCA 임베딩 시스템 초기화")
        print("="*70)
        
        # 1. PCA 모델 로드
        print(f"📥 PCA 모델 로딩: {settings.PCA_MODEL_PATH}")
        self.pca = joblib.load(settings.PCA_MODEL_PATH)
        print(f"✅ PCA 로드 완료!")
        print(f"   입력 차원: {self.pca.n_features_in_}")
        print(f"   출력 차원: {self.pca.n_components_}")
        
        # 2. 디바이스 설정
        self.device = "cuda" if (settings.USE_GPU and torch.cuda.is_available()) else "cpu"
        print(f"📱 Device: {self.device}")
        
        # 3. Alibaba GTE 모델 로드
        print(f"📥 임베딩 모델 로딩: {settings.EMBEDDING_MODEL}")
        print(f"⚠️  7B 모델 로딩 중... 시간이 걸립니다")
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(
            settings.EMBEDDING_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"✅ 모델 로드 완료!")
        
        print("="*70)
        print("✅ 초기화 완료!")
        print("="*70)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 문장 임베딩 생성"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _get_gte_embedding(self, text: str) -> np.ndarray:
        """
        Alibaba GTE 임베딩 생성
        - 벡터 DB 생성 시와 동일한 방식
        """
        # 토크나이징
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 임베딩 생성
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # numpy로 변환
        embedding = embeddings.cpu().numpy()[0]
        
        # 정규화
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _adjust_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """
        임베딩 차원을 PCA 입력 차원에 맞춤
        """
        target_dim = self.pca.n_features_in_
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        
        if current_dim > target_dim:
            # 자르기
            return embedding[:target_dim]
        else:
            # 패딩
            return np.pad(
                embedding,
                (0, target_dim - current_dim),
                mode='constant'
            )
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리 임베딩
        
        Args:
            text: 쿼리 텍스트 (예: "부산 실내 놀이터")
        
        Returns:
            PCA 변환된 512차원 벡터
        """
        # 1. Alibaba GTE 임베딩 (4096차원)
        embedding = self._get_gte_embedding(text)
        
        # 2. 차원 조정
        embedding = self._adjust_dimension(embedding)
        
        # 3. PCA 변환 (512차원)
        pca_embedding = self.pca.transform([embedding])[0]
        
        return pca_embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서 임베딩
        
        Args:
            texts: 문서 리스트
        
        Returns:
            PCA 변환된 512차원 벡터 리스트
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                print(f"   임베딩 진행: {i+1}/{len(texts)}")
            
            embedding = self._get_gte_embedding(text)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # 차원 조정
        adjusted_embeddings = []
        for emb in embeddings:
            adjusted_embeddings.append(self._adjust_dimension(emb))
        adjusted_embeddings = np.array(adjusted_embeddings)
        
        # PCA 변환
        pca_embeddings = self.pca.transform(adjusted_embeddings)
        
        return pca_embeddings.tolist()

# 싱글톤 인스턴스
pca_embeddings = PCAEmbeddings()