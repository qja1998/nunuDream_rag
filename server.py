from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, List
import json

from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

print("import 완료")

# loader = CSVLoader("./data/fss_data.csv")
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
# splits = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # 금융용어 추출 사전 준비
# loader_fin = CSVLoader("./data/fin_word.csv")
# documents_fin = loader_fin.load()

# embeddings_fin = OpenAIEmbeddings()
# vectorstore_fin = FAISS.from_documents(documents=documents_fin, embedding=embeddings_fin)



# system_message_dict = {
#     "예금/적금 추천": """
#     당신은 친절한 금융 전문가입니다. 주어진 {context}를 바탕으로 사용자에게 적절한 금융 상품을 추천하는 역할을 합니다.
#     - {context}에서 [fin_prdt_cd]이 같은 optionList를 찾아 다양한 정보를 비교하세요.
#     - 하나의 상품으로 목표 달성이 어려우면 **여러 개**의 다양한 상품을 추천하여 목표를 달성하도록 합니다.
#     - {question}에서 금액,기간이 모두 주어지지 않으면 필요한 정보를 요구합니다.
#     - {context}에서 추출한 회사명과 상품명을 사용합니다.
#     - 금리와 같은 중요한 정보를 명확하게 제공합니다.
#     - 특이사항을 제공합니다.
#     - 제품을 요약합니다.
#     사용자의 상황을 분석하고,
#     예금과 적금을 적절하게 사용하여 기간 내에 금액을 만드는 전략 시나리오를 만듭니다.
#     이후 해당 시나리오를 수행할 수 있는 금융 상품을 추천합니다.
#     """,
#     "계산": """
#     당신은 유능한 금융 전문가입니다. 사용자의 {question}이 적금인지 예금인지 판단하고 기간과 금액, 이자율에 따라 얻을 수 있는 이득을 계산합니다.
#     예금은 예치금과 기간이 주어질 것이고, 적금은 월 저축금과 기간이 주어질 것입니다.
#     - 각 이득을 계산하는 과정을 수식과 함께 자세히 서술
#     - 이자율 대신 상품명이 있다면 {context}에서 찾아 계산
#     """,
#     "기타": "당신은 금융 전문가입니다. {question}에 대해 {context}를 기반으로 적절히 답변하세요."
# }

# def get_purpose(question):
#     prompt = """
#         Q의 목적을 다음 중 선택하여 출력하시오.
#         - 예금/적금 추천
#         - 계산
#         - 기타
#         오직 이 중에 하나를 선택해서 그 것만을 출력합니다.
#         Q.
#     """
#     result = llm.invoke(prompt + question)
#     return result.content

# def qna(question):
#     purpose = get_purpose(question).replace('-', '').strip()
#     print('목적:', purpose)
#     system_message = system_message_dict[purpose]

#     # Define the prompt template
#     qa_prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=system_message
#     )

#     # Set up RetrievalQA with the correct combination of LLM and retriever
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,  # Pass the llm directly
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#         chain_type_kwargs={"prompt": qa_prompt}
#     )

#     result = chain({'query': question})
#     return result['result']


# parser = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# def get_prdt(question):
#     parser = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     response = parser.invoke("Q:" + question + "\nQ에서 은행명과 예적금을 모두 추출\n은행명1-예적금1,은행명2-예적금2").content
#     print(response)
#     response = response.split(',')
#     result = []
#     for row in response:
#         bank, prdt = row.split('-')
#         result.append([{'bank':bank}, {'prdt':prdt}])
#     return result

# def get_fin(question):
#     system_message = """
#         {question}에서 등장한 용어 중 {context}에 없는 단어를 삭제합니다.
#         format: 용어1,용어2,용어3
#     """
#     qa_prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=system_message
#     )

#     # Set up RetrievalQA with the correct combination of LLM and retriever
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,  # Pass the llm directly
#         retriever=vectorstore_fin.as_retriever(),
#         memory=memory,
#         chain_type_kwargs={"prompt": qa_prompt}
#     )

#     result = chain({'query': question})
#     return result['result'].split(',')



app = FastAPI()
# uvicorn server:app --host 0.0.0.0 --port 8000



class CahtItem(BaseModel):
    answer: str

# Read - 모든 아이템 조회
@app.get("/chat/")
async def get_chat(query, response_model=CahtItem):
    global memory

    # answer = CahtItem(
    #     answer=qna(query)
    # )
    # answer = JSONResponse(content=jsonable_encoder({'answer':qna(query)}))
    
    answer = {
        'answer':'a'
    }
    return answer


# class PRDTItem(BaseModel):
#     prdt: str
#     bank: str

# class PRDTItems(BaseModel):
#     prdts: List[PRDTItem]

# @app.get("/extractPrdt/")
# async def get_extrct_prdt(query):
#     # prdts = [PRDTItem(prdt=item['prdt'], bank=item['bank']) for item in get_prdt(query)]
#     # answer = PRDTItem(
#     #     prdts=prdts
#     # )
#     answer = JSONResponse(content=jsonable_encoder({'prdts':get_prdt(query)}))

#     return answer





# @app.get("/extractFin/")
# async def get_extrct_fin(query):
#     answer = {
#         'fins': get_fin(query)
#     }
#     return json.dumps(answer, ensure_ascii=False)