import json, requests, time

API_KEY = ## insert API key
BASE_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency{}'

HEADERS = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY ,
}


class CoinmarketCap:
    def __init__(self):
        self.cache = {}
        self.cache_time = 5


    def request(self, url, parameters):
        key = (url, json.dumps(parameters))

        if key not in self.cache or time.time() - self.cache[key][0] > self.cache_time:
            r = requests.get(url=url, params=parameters, headers=HEADERS)
            self.cache[key] = (time.time(), json.loads(r.text))

        return self.cache[key][1]


    def get_coin_map(self):
        parameters = {
            'start': '1',
            'limit': '5000',
            'convert': 'USD'
        }
        j = self.request(BASE_URL.format('/listings/latest') ,parameters)
        coins = { x['symbol'] : x['id'] for x in j['data']}
        return coins

    def get_quote(self, symbol, base_symbol):
        try:
            parameters = {
                'symbol' : symbol,
                'convert' : base_symbol
            }
            # r = requests.get(url=BASE_URL.format('/quotes/latest'), params=parameters, headers=HEADERS)
            # j = json.loads(r.text)

            j = self.request(BASE_URL.format('/quotes/latest'), parameters)

            return j['data'][symbol]['quote'][base_symbol]['price']
        except:
            return -1

coin_market_cap = CoinmarketCap()

def get_quote(symbol, base):
    return coin_market_cap.get_quote(symbol, base)
