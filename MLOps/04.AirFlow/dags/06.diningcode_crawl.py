import requests

headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'ko,en;q=0.9,en-US;q=0.8',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Origin': 'https://www.diningcode.com',
    'Referer': 'https://www.diningcode.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0',
    'sec-ch-ua': '"Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
}

data = {
    'query': '합정동',
    'addr': '',
    'keyword': '',
    'order': 'r_score',
    'distance': '',
    'rn_search_flag': 'on',
    'search_type': 'poi_search',
    'lat': '',
    'lng': '',
    'rect': '',
    's_type': '',
    'token': '',
    'mode': 'poi',
    'dc_flag': '1',
    'page': '2',
    'size': '20',
}

response = requests.post('https://im.diningcode.com/API/isearch/', headers=headers, data=data)


print(response.text)