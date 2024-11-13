# ssafy-rag-chatbot

### PostgreSQL 설치 및 설정

#### Linux (Ubuntu)

1. PostgreSQL 설치

```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

2. 서비스 상태 확인

```bash
sudo service postgresql status
```

#### PostgreSQL 데이터베이스 설정

1. PostgreSQL 접속

```bash
sudo -i -u postgres
psql
```

2. 데이터베이스 생성

```sql
CREATE DATABASE ssafyrag;
```

3. 사용자 생성 및 권한 부여

```sql
CREATE USER ssafyuser WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ssafyrag TO ssafyuser;
ALTER USER ssafyuser WITH SUPERUSER;
```

4. exit로 빠져나오기

5. PostgreSQL 서버 시작/종료 (Linux)

```
sudo service postgresql start   # 서버 시작
sudo service postgresql stop    # 서버 종료
sudo service postgresql restart # 서버 재시작
```

PostgreSQL의 기본 포트 번호는 5432입니다.

======

```
poetry install
poetry shell
streamlit run app.py
```

document_store.py 돌려서 PGVector로 NewsDocumentStore 임베딩해서 저장하고,

chatbot.py에서 streamlit으로 자유롭게 대화하면 됨
