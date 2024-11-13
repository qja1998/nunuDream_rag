from langchain.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
from preprocessing import load_json_files_and_merge

class NewsDocumentStore:
    """
    뉴스 기사를 위한 문서 저장 및 검색 시스템
    PGVector를 백엔드로 사용하여 벡터 검색을 구현합니다.
    """
    
    def __init__(self):
        # 환경 변수 로드
        load_dotenv()
        
        # 데이터베이스 연결 문자열 생성
        self.connection_string = PGVector.connection_string_from_db_params(
            driver=os.getenv("DB_DRIVER", "psycopg2"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "vectordb"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "1234")
        )
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 벡터 저장소 초기화
        self.vectorstore = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name="news_documents"
        )
    
    def process_news_article(self, article: Dict) -> Dict:
        """
        뉴스 기사 데이터를 처리하여 문서와 메타데이터로 변환합니다.
        
        Args:
            article: 뉴스 기사 데이터 딕셔너리
            
        Returns:
            처리된 문서와 메타데이터
        """

        # 본문 텍스트 처리
        # content = f"{article['title']}\n\n{article.get('subtitle', '')}\n\n{article['content']}"
        content = ""
        for val in article.values():
            print(val)
            content += f"{val}\n\n"
        
        # 메타데이터 구성
        metadata = {
            'kor_co_nm': article['kor_co_nm'],
            'fin_prdt_nm': article.get('fin_prdt_nm', ''),
            'spcl_cnd': article.get('spcl_cnd', ''),
            'join_deny': article.get('join_deny', ''),
            'join_member': article.get('join_member', ''),
            'etc_note': article.get('etc_note', ''),
            'max_limit': article.get('max_limit', ''),
            'is_diposit': article.get('is_diposit', ''),
            'intr_rate_type': article.get('intr_rate_type', ''),
            'intr_rate_type_nm': article.get('intr_rate_type_nm', ''),
            'save_trm': article.get('save_trm', ''),
            'intr_rate': article.get('intr_rate', ''),
            'intr_rate2': article.get('intr_rate2', ''),
            'rsrv_type': article.get('rsrv_type', ''),
            'rsrv_type_nm': article.get('rsrv_type_nm', ''),
            'processed_date': datetime.now().isoformat()
        }
        
        return {
            'content': content,
            'metadata': metadata
        }
    
    def add_news_articles(self, articles: List[Dict]) -> None:
        """
        뉴스 기사들을 벡터 저장소에 추가합니다.
        
        Args:
            articles: 뉴스 기사 데이터 리스트
        """
        documents = []
        for article in articles:
            # try:
                processed = self.process_news_article(article)
                documents.append(
                    Document(
                        page_content=processed['content'],
                        metadata=processed['metadata']
                    )
                )
            # except Exception as e:
            #     print(e)
            #     pass

        print(f"처리된 기사 개수: {len(documents)}")
        
        # 벡터 저장소에 문서 추가
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        유사도 기반 뉴스 기사 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            filter_dict: 메타데이터 기반 필터링 조건 (선택사항)
            
        Returns:
            유사한 뉴스 기사들의 리스트
        """
        # 유사도 검색 수행
        docs = self.vectorstore.similarity_search_with_score(
            query, 
            k=k,
            filter=filter_dict
        )
        
        # 결과 포매팅
        results = []
        for doc, score in docs:
            results.append({
                'title': doc.metadata.get('title', ''),
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity': 1 - score  # 점수를 유사도로 변환
            })
            
        return results
    
    @classmethod
    def from_existing(cls, collection_name: str = "news_documents"):
        """
        기존 컬렉션을 사용하여 NewsDocumentStore 인스턴스를 생성합니다.
        
        Args:
            collection_name: 사용할 컬렉션 이름
            
        Returns:
            NewsDocumentStore 인스턴스
        """
        instance = cls()
        instance.vectorstore = PGVector(
            connection_string=instance.connection_string,
            embedding_function=instance.embeddings,
            collection_name=collection_name
        )
        return instance
    
    def delete_collection(self):
        """
        현재 컬렉션을 삭제합니다.
        """
        self.vectorstore.delete_collection()
        print(f"Collection '{self.vectorstore.collection_name}' has been deleted successfully.")

# 사용 예시
if __name__ == "__main__":
    # 뉴스 문서 저장소 초기화
    news_store = NewsDocumentStore.from_existing("news_documents")
    
    merged_data_list = load_json_files_and_merge('data')[:3000]
    print(len(merged_data_list))
    
    # # 100개 랜덤 샘플링
    # merged_data_list = random.sample(merged_data_list, 30)
    
    # 문서 추가
    news_store.add_news_articles(merged_data_list)
    
    # 검색 예시
    results = news_store.similarity_search(
        "라면 좋아해?",
        k=3,
    )
    
    # 결과 출력
    for i, result in enumerate(results, 1):
        print(f"\n=== 검색 결과 {i} ===")
        print(f"제목: {result['title']}")
        print(f"출처: {result['metadata']['source_site']}")
        print(f"작성일: {result['metadata']['write_date']}")
        print(f"유사도: {result['similarity']:.4f}")
        print(f"내용 미리보기: {result['content'][:200]}...")