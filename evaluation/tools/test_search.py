from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import requests
import re
from argparse import Namespace
import os
import json

BING_API_KEY = os.environ.get("BING_API_KEY", "")
BING_ENDPOINT = os.environ.get("BING_ENDPOINT", "")

def langsearch(query, top_n=5, bing_subscription_key=BING_API_KEY, bing_endpoint=BING_ENDPOINT):
    payload = json.dumps({
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": top_n
    })
    headers = {
    'Authorization': bing_subscription_key,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", bing_endpoint, headers=headers, data=payload)
    # print(response.text)
    # print(response.json())
    return response.json()

def deep_search_snippet(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key=BING_API_KEY, bing_endpoint=BING_ENDPOINT):
    # 根据函数参数构建 args
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  # 使用函数参数
        use_jina=use_jina,  # 使用函数参数
        jina_api_key=jina_api_key,  # 使用函数参数
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  # 使用函数参数
        bing_endpoint=bing_endpoint,  # 使用函数参数
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
    # print(args)
    search_cache = {}
    url_cache = {}

    question = search_query

    try:
        # 调用必应搜索API
        results = langsearch(question,bing_subscription_key= args.bing_subscription_key,bing_endpoint= args.bing_endpoint) 
        search_cache[question] = results
    except Exception as e:
        print(f"Error during search query '{question}': {e}")
        results = {}
    print(results)
    # 提取相关信息并限制结果数量
    # relevant_info = extract_relevant_info(results["data"])
    # print(relevant_info)
    # print("--------------------------------Search Bing Result--------------------------------")

    # result = ""
    # for info in relevant_info:
    #     # info['snippet'] = formatted_documents
    #     snippet = info['snippet']
    #     clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags
    #     result+=clean_snippet

    extracted_info = results

    return extracted_info
def batch_search(query, top_n=5):
    if len(query) == 0:
        return 'invalid query'

    url = f'http://183.174.229.164:1243/batch_search'
    if isinstance(query, str):
        query = [query]
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    
    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            curr_result += f"{line['contents']}\n\n"
    result_list.append(curr_result.strip())
    
    return result_list

if __name__ == "__main__":
    query = "2023年拳头游戏Pride Month合作音乐艺术家 全球总决赛相关游戏角色?"
    top_n = 5
    result = deep_search_snippet(query)
    print(result)
