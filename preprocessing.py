# 데이터 소스 경로
import json
import os
import pandas as pd
import csv
import re

def load_json_files_and_merge(base_directory):
    all_data = []
    # 디렉토리 내의 모든 JSON 파일 순회
    for filename in os.listdir(base_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(base_directory, filename)
            
            # JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if 'SJML' in data and 'text' in data['SJML']:
                    for text_item in data['SJML']['text']:
                        all_data.append(text_item)
                
            # 데이터 리스트에 추가
            all_data.append(data)
    
    return all_data

def load_json_files_and_merge(base_directory):
    all_data = []
    # 디렉토리 내의 모든 JSON 파일 순회
    df = pd.read_csv("data/fss_data.csv")
    keys = df.columns
    for data in df.values:
        # print(data)
        contnets = []
        
        contnets = {keys[i] : t for i, t in enumerate(data)}
        # print(text)
        all_data.append(contnets)
    # print(all_data[0])
    return all_data

def clean_text(base_directory):

    # 디렉토리 내의 모든 JSON 파일 순회
    for filename in os.listdir(base_directory):
        if filename.endswith('.csv') and not filename.startswith('clean'):
            with open(os.path.join(base_directory, filename), 'r', encoding='utf-8') as infile, open(os.path.join(base_directory, 'cleand_'+filename), 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # 특수문자를 제거: 알파벳, 숫자, 공백을 제외한 모든 문자 제거
                    cleaned_line = re.sub(r'[^\w\s,]', '', line)
                    # 정제된 데이터를 출력 파일에 기록
                    outfile.write(cleaned_line)

# def load_json_files_and_merge(base_directory):
#     all_data = []
#     # 디렉토리 내의 모든 JSON 파일 순회
#     for filename in os.listdir(base_directory):
#         if filename.endswith('.csv') and filename.startswith('clean'):
#             df = pd.read_csv(os.path.join(base_directory, filename), sep='|', quoting=csv.QUOTE_NONE, encoding='utf-8')
#             json_str = df.to_json(orient='records', force_ascii=False)
#             # print(json.loads(json_str)[0].keys())
#             for row in json.loads(json_str):
#                 all_data.append(row['companytitlelinkpublishedcategorycategory_strreporterarticle'])
#     return all_data



if __name__ == "__main__":
    base_directory = 'data'

    # clean_text(base_directory)

    # 데이터 로드 및 병합 (이 부분을 크롤링으로 대체해도 좋습니다)
    merged_data_list = load_json_files_and_merge(base_directory)

    print(len(merged_data_list))
    print(merged_data_list[0])