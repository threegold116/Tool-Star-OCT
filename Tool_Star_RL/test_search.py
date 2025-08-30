
import requests
def batch_search(query, top_n=5):
        if len(query) == 0:
            return 'invalid query'

        url = f'http://0.0.0.0:8008/batch_search' #your local search path
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
print(batch_search("Jackie chen"))