import os
from document_store import NewsDocumentStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage, BaseChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import requests
import re
from bs4 import BeautifulSoup as bs

# Load environment variables
load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        # print(self.text, end="\r")

def initialize():
    """Initialize document store and language model."""
    doc_store = NewsDocumentStore.from_existing("news_documents")
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    return doc_store, llm

def search_documents(doc_store, query: str, k: int = 3):
    """Search for relevant documents."""
    try:
        return doc_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []

def extract_company(query: str, llm) -> str:
    """Extract company name from user input."""
    system_prompt = "Extract the company name from the user's input."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Input: {query}")
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error during company name extraction: {str(e)}")
        return "Error extracting company name."

def clean(text):
    i = text.find('글자수')
    text = text[:i]
    return text.strip()

def get_company_recruit(search_keyword):
    base_url = "https://www.jobkorea.co.kr/starter/PassAssay"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(f"{base_url}?schTxt={search_keyword}", headers=headers)
    soup = bs(response.text, 'html.parser')
    items = [item.select('.txBx') for item in soup.select('#container > div.stContainer > div.starListsWrap.ctTarget > ul > li')]

    if not items or not items[0]:
        return None
    
    qna_texts = []
    for item in items:
        item_id = re.findall("\d{6}", item[0].select('a')[0].attrs['href'])[0]
        response_ans = requests.get(f"{base_url}/View/{item_id}", headers=headers)
        soup_ans = bs(response_ans.text, 'html.parser')
        
        questions = [q.select_one('.tx').text for q in soup_ans.select('#container > div.selfQnaWrap > dl > dt')]
        answers = [clean(a.select_one('.tx').text) for a in soup_ans.select('#container > div.selfQnaWrap > dl > dd')]
        
        for q, a in zip(questions, answers):
            qna_texts.append(f"Question: {q} \nAnswer: {a}")
    
    return "\n".join(qna_texts)

def generate_answer(query: str, relevant_docs: list, llm, stream_handler):
    """Generate answer using GPT model based on documents and user query."""
    # company_name = extract_company(query, llm).replace(' ', '')
    # print(f"Company name extracted: {company_name}")
    
    # company_qna = get_company_recruit(company_name)
    # if company_qna is None:
    #     context = "Related document information:\n"
    # else:
    #     context = f"QnA: {company_qna}\nRelated document information:\n"

    context = "Related document information:\n"
    for i, doc in enumerate(relevant_docs, 1):
        print(doc)
        context += f"{i}. {doc['content']}\n"

    system_prompt = """
    당신은 친절한 금융 전문가입니다. 당신의 역할은 주어진 정보를 바탕으로 사용자에게 작절한  금융 상품을 추천하는 것입니다. {fin_prdt_cd}이 같은 optionList를 찾아 다양한 정보를 비교하세요
    다음 예시와 같이 다양한 상품을 추천하세요. {}는 내용이 들어갈 공간이며 내용을 채운 후에는 삭제합니다.
    Q: info: {월급}, {예치금}, {기간}, {월 적립금}
    저에게 적절한 {is_deopsit} 상품을 추천해주세요.
    A: {금액}과 {기간}에 따라 추천드릴 {is_deposit} 상품은 {kor_co_nm}의 {fin_prdt_nm}입니다. 해당 상품은 {intr_rate}의 높은 금리를 가지고 있으며, {fin_prdt_info-_summation}과 같은 특징을 가지고 있어 사용자님에게 적절하다고 판단됩니다.
    {intr_rate}의 이자율로 계산을 했을 때 {compute_benefit} 정도의 이득을 볼 수 있습니다.
    가입 전에 {etc_note}과 같은 특이사항이 있으니 확인하시기 바랍니다.
    """
    


    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_prompt}",
            ),
            # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),  # 사용자 입력을 변수로 사용
        ]
    )
    runnable = prompt | llm  # 프롬프트와 모델을 연결하여 runnable 객체 생성

    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = {}  # 세션 기록을 저장할 딕셔너리

    # 세션 ID를 기반으로 세션 기록을 가져오는 함수
    def get_session_history(session_ids: str) -> BaseChatMessageHistory:
        print(session_ids)
        if session_ids not in store:  # 세션 ID가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            store[session_ids] = ChatMessageHistory()
        return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


    with_message_history = (
        RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
            runnable,  # 실행할 Runnable 객체
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="history",  # 기록 메시지의 키
        )
    )

    # messages = [
    #     SystemMessage(content=system_prompt),
    #     HumanMessage(content=f"문서 정보에 따라 금융 상품을 추천하세요.\n\n{context}\n\nQuestion: {query}")
    # ]
    
    # try:
    # response = llm(messages, callbacks=[stream_handler])
    response = with_message_history.invoke(
        # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
        {"system_prompt":system_prompt, "input": f"문서 정보에 따라 금융 상품을 추천하세요.\n\n{context}\n\nQuestion: {query}"},
        # 설정 정보로 세션 ID "abc123"을 전달합니다.
        config={"configurable": {"session_id": "abc123"}},
    )
    return response.content.strip()
    # except Exception as e:
    #     print(f"Error generating answer: {str(e)}")
    #     return "Sorry, there was an error generating an answer."

def main():
    doc_store, llm = initialize()
    chat_history = []
    
    print("Welcome to ChuiBBOt! Ask your questions or type 'exit' to quit.")
    
    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            break
        
        chat_history.append({"role": "user", "content": query})
        
        print("Searching for relevant documents...")
        results = search_documents(doc_store, query)
        
        print("Generating answer...")
        stream_handler = StreamHandler()
        answer = generate_answer(query, results, llm, stream_handler)
        chat_history.append({"role": "assistant", "content": answer})
        
        print("\nChuiBBOt's Answer:")
        print(answer)
        
        print("\nDo you want to continue? (Type 'exit' to quit or ask another question.)")

if __name__ == "__main__":
    main()
