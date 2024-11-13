import streamlit as st
from document_store import NewsDocumentStore
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
import requests
import re
from bs4 import BeautifulSoup as bs

# 환경 변수 로드
load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.message_placeholder = container.empty()

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.message_placeholder.markdown(self.text)

def initialize_session_state():
    """Streamlit 세션 상태 초기화"""
    if 'doc_store' not in st.session_state:
        # 기존 collection을 사용하여 NewsDocumentStore 초기화
        st.session_state.doc_store = NewsDocumentStore.from_existing("news_documents")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            streaming=True  # 스트리밍 활성화
        )

def search_documents(query: str, k: int = 3):
    """문서 검색 함수"""
    try:
        return st.session_state.doc_store.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
        return []

def extract_company(query: str) -> str:
    """GPT를 사용하여 답변 생성"""
    # 시스템 프롬프트 구성
    system_prompt = """당신을 사용자의 입력에서 회사명을 추출해야 합니다. 적절한 회사명을 추출하여 회사명만 답변하세요."""
    

    # 메시지 구성
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"입력: {query}")
    ]
    
    try:
        # GPT로 스트리밍 답변 생성 (invoke 메소드 사용)
        response = st.session_state.llm.invoke(
            messages
        )
        return response.content
    
    except Exception as e:
        st.error(f"회사명 추출 중 오류 발생: {str(e)}")
        return "죄송합니다. 회사명 추출 중에 문제가 발생했습니다."

def clean(text):
    i = text.find('글자수')
    text = text[:i]
    return text.strip()


def get_company_recruit(search_keyword):
    base_url = "https://www.jobkorea.co.kr/starter/PassAssay"
    headers={'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}
    response = requests.get(f"{base_url}?schTxt={search_keyword}", headers=headers)
    soup = bs(response.text,'html.parser')
    items = []
    for _, item in enumerate(soup.select('#container > div.stContainer > div.starListsWrap.ctTarget > ul > li')):
        items.append(item.select(f'.txBx'))
    
    soup_ans_list = []
    if not items[0]:
        return None
    for item in items:
        item_id = re.findall("\d{6}", item[0].select('a')[0].attrs['href'])[0]
        response_ans = requests.get(f"{base_url}/View/{item_id}?Page=1&OrderBy=0&schTxt={search_keyword}&FavorCo_Stat=0&Pass_An_Stat=0", headers=headers)
        soup_ans_list.append(bs(response_ans.text,'html.parser'))

    q_list = []
    a_list = []
    for soup_ans in soup_ans_list:
        q_list += [q.select_one('.tx').text for q in soup_ans.select('#container > div.stContainer > div.selfQnaWrap > dl > dt')]
        a_list += [clean(a.select_one('.tx').text) for a in soup_ans.select('#container > div.stContainer > div.selfQnaWrap > dl > dd')]
    
    qna = ""
    for q, a in zip(q_list, a_list):
        qna += f"질문: {q} \n답변: {a}\n"
    
    return qna

def generate_answer(query: str, relevant_docs: list, stream_handler) -> str:
    """GPT를 사용하여 답변 생성"""
    # 시스템 프롬프트 구성

    company_name = extract_company(query).replace(' ', '')
    print(company_name)
    company_qna = get_company_recruit(company_name)
    # print(company_qna)

    system_prompt = """당신은 구직자의 질문에 대해 기업의 정보를 친절하고 전문적으로 답변하는 AI 어시스턴트입니다.
    제공된 문서 정보를 바탕으로 답변을 생성하되, 문서 내용을 인용합니다.
    QnA는 이미 취업을 성공한 사용자들의 자기소개서입니다. 분석해서 함께 제공하세요.
    답변은 한국어로 작성하며, 취업을 위한 자기소개서/면접에 도움이 될 내용 정확하고 자세하게 알려주세요.
    user: {company}
    you: {company}는 {context}한 상황입니다.
        **취업 준비를 위한 팁**: {company}와 같은 기업에 지원할 때는 {recommend} 한 것들을 신경쓰는 것이 좋습니다.
        **성공 사례**:
        {q}에 대한 문항에,
        합격자들은 {a}라고 답했습니다.
        그러므로 당신은 {solution}을 염두에 두고 준비할 수 있습니다."""
    
    # 관련 문서 정보 구성
    if company_qna is None:
        context = f"관련 문서 정보:\n"
    else:
        context = f"QnA: {company_qna}\n관련 문서 정보:\n"

    for i, doc in enumerate(relevant_docs, 1):
        context += f"{i}. {doc['content']}\n"
    
    # 메시지 구성
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"다음 문서 정보를 참고하여 기업에 대한 최신 정보와 동향을 요약 설명해주세요.\n\n{context}\n\n질문: {query}")
    ]
    
    try:
        # GPT로 스트리밍 답변 생성 (invoke 메소드 사용)
        response = st.session_state.llm(
            messages,
            callbacks=[stream_handler]
        )
        return response.content
    except Exception as e:
        st.error(f"답변 생성 중 오류 발생: {str(e)}")
        return "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다."

def main():
    st.set_page_config(
        page_title="AI 챗봇",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # 메인 채팅 영역과 사이드바 레이아웃 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("ChuiBBOt")
        
        # 채팅 컨테이너 생성
        chat_container = st.container()
        with chat_container:
            # 채팅 히스토리 표시
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # 사용자 입력 영역
        if question := st.chat_input("질문을 입력하세요"):
            # 사용자 메시지 표시
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # 관련 문서 검색
                with st.spinner("관련 정보 검색 중..."):
                    results = search_documents(question)

                # 사이드바 업데이트를 위한 상태 저장
                st.session_state.latest_results = results
                
                # GPT 답변 생성 및 표시
                with st.chat_message("assistant"):
                    stream_handler = StreamHandler(st.empty())
                    answer = generate_answer(question, results, stream_handler)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
    # 사이드바: 관련 문서 정보 표시
    with col2:
        st.sidebar.title("📑 관련 문서")
        # 현재 질문 표시
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            last_user_message = next((msg for msg in reversed(st.session_state.chat_history) 
                                    if msg["role"] == "user"), None)
            if last_user_message:
                st.sidebar.markdown("**현재 질문:**")
                st.sidebar.info(last_user_message["content"])
                
        # 검색 결과 표시
        if 'latest_results' in st.session_state and st.session_state.latest_results:
            for i, result in enumerate(st.session_state.latest_results, 1):
                with st.sidebar.expander(f"관련 문서 {i} (유사도: {result['similarity']:.2%})"):
                    st.markdown("**내용:**")
                    st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                    st.markdown("**메타데이터:**")
                    for key, value in result['metadata'].items():
                        st.write(f"- {key}: {value}")
        else:
            st.sidebar.info("관련 문서가 없습니다.")
        
        # 대화 초기화 버튼
        if st.sidebar.button("대화 내용 초기화", type="secondary"):
            st.session_state.chat_history = []
            if 'latest_results' in st.session_state:
                del st.session_state.latest_results
            st.rerun()

if __name__ == "__main__":
    main()