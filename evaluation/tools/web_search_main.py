import json
import sys
import os

from zmq import Errno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urllib.parse import urljoin
import time 
from argparse import Namespace

from traitlets import default
from tools.bing_search import bing_web_search
from tools.bing_search import extract_relevant_info
from tools.lang_search import langsearch
import re
#THREEGOLDCHANGE:添加读取cache操作
default_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cache", "search_cache.json")
search_cache_file = os.environ.get("SEARCH_CACHE_FILE", default_cache_file)
search_cache = {}
if os.path.exists(search_cache_file):
    try:
        print(f"load search cache from {search_cache_file}")
        with open(search_cache_file, "r", encoding='utf-8') as f:
            search_cache = json.load(f)
    except Exception as e:
        cache_timestamp_files = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"cache"))
        cache_timestamp_files.remove("search_cache.json")
        cache_timestamp_files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        latest_search_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cache",cache_timestamp_files[-1])
        print(f"load latest search cache from {latest_search_cache_file}")
        with open(latest_search_cache_file, "r", encoding='utf-8') as f:
            search_cache = json.load(f)
else:
    os.makedirs(os.path.dirname(search_cache_file), exist_ok=True)
#THREEGOLDCHANGE:添加读取cache操作
def deep_search(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="xxxxx", bing_endpoint="xxxxx/search",search_engine="langsearch"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  
        bing_endpoint=bing_endpoint,  
        eval=False,
        seed=1742208600,
        api_base_url='xxxxx',  
        model_name='search-agent',
        concurrent_limit=200
    )
    global search_cache

    question = search_query
    #THREEGOLDCHANGE:
    if question in search_cache:
        results = search_cache[question]
        print(f"load search cache from {search_cache_file} for question {question}")
    #THREEGOLDCHANGE:
    else:
        try:
            if search_engine == "bing": #TODO:其他search engines通过环境变量获取endpoint和key
                results = bing_web_search(question, args.bing_subscription_key, args.bing_endpoint) 
            elif search_engine == "langsearch": 
                results = langsearch(question,top_n=args.top_k)
            elif search_engine == "google":
                raise NotImplementedError("Google search is not implemented yet")
            else:
                raise NotImplementedError(f"Search engine {search_engine} is not implemented yet")
            if len(results.keys())>0:
                search_cache[question] = results
        except Exception as e:
            print(f"Error during search query '{question}': {e}")
            results = {}
    
    if search_engine == "bing":
        relevant_info = extract_relevant_info(results)[:args.top_k]
    elif search_engine == "langsearch":
        relevant_info = extract_relevant_info(results)
    else:
        raise NotImplementedError(f"Search engine {search_engine} is not implemented yet")
    print("--------------------------------get bing search result--------------------------------")

    result = ""
    for info in relevant_info:
        snippet = info['snippet']
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  
        result+=clean_snippet

    extracted_info = result

    return extracted_info


if __name__ == "__main__":

    extracted_info = ''
    
    
    question = "What is the capital of France?"

    result = deep_search(
        question
        ,top_k=2
    )
    print('-------------------------------------')
    print(result)
    print('-------------------------------------')
    
    