import requests

# URL for your local FastAPI server
url = "http://127.0.0.1:8000/search"

# Example payload
payload = {
    "query": "1+1=?",
    "top_n": 5,
    "return_score": True
}

# Send POST request
response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)
