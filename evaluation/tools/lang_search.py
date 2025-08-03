import requests
import os
import json
import time

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
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.request("POST", bing_endpoint, headers=headers, data=payload)
            response.raise_for_status()  # Raise exception if the request failed
            return response.json()["data"]
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Langs Search request failed after {max_retries} retries")
                return {}
        time.sleep(1)