from web3 import Web3
import json, yaml, CMC, requests, time
from functools import lru_cache

class Requestbase:
    def __init__(self, chain, store_file, rpc=None, abi_end_point=None):
        self.config = store_file
        self.store_file = store_file[chain]
        self.rpc = self.store_file['rpc_url']
        self.w3 = Web3(Web3.HTTPProvider(self.rpc))
        self.abi_end_point = self.store_file['abi_end_point']
        self.chain_addresses = self.store_file['addresses']

        self.abi_end_point_text = self.store_file['abi_end_point_text']
        try:
            self.cache = store_file['cache']
        except:
            self.cache = {'requests': {}, 'cache': {}}

        self.lending_address = self.store_file['lending_contract_address']
        self.lending_abi_address = self.store_file['lending_contract_abi_address']
        self.incentives_address = self.store_file['incentives_contract_address']
        self.incentives_abi_address = self.store_file['incentives_contract_abi_address']

    def cached_request(self, url):
        if url in self.cache['requests']:
            return self.cache['requests'][url]

        r = requests.get(url)
        self.cache['requests'][url] = r.text
        return r.text

    @lru_cache
    def contractv2(self, token_address, abi_address=None):
        try:
            time.sleep(.5)
            API_ENDPOINT = self.abi_end_point_text.format(token_address)
            try:
                abi = json.loads(self.config[API_ENDPOINT])
            except:
                abi = json.loads(self.cached_request(url=API_ENDPOINT))
        except Exception as e:
            print('Failed to get ABI', e, token_address)
            return None

        return self.w3.eth.contract(address=self.w3.toChecksumAddress(token_address),
                                    abi=abi)

    def backup(self):
        self.cache['cache'] = self.cache['requests']
        try:
            if self.config[list(self.cache['cache'].keys())[0]]:
                return print('ABI already exists.')
        except:
            with open('abi_config.yaml', 'a') as w:
                yaml.dump(self.cache['cache'], w, allow_unicode=True)
        self.cache = {'requests': {}, 'cache': {}}
        return True

    @lru_cache
    def lending_contract(self):
        API_ENDPOINT = self.abi_end_point_text.format(self.lending_abi_address)
        try:
            abi = json.loads(self.config[API_ENDPOINT])
        except:
            abi = json.loads(self.cached_request(url=API_ENDPOINT))

        contractLending = self.w3.eth.contract(
            address=self.w3.toChecksumAddress(self.lending_address),
            abi=abi
        )
        return contractLending

    @lru_cache
    def incentives_contract(self):
        API_ENDPOINT = self.abi_end_point_text.format(self.incentives_abi_address)
        try:
            abi = json.loads(self.config[API_ENDPOINT])
        except:
            abi = json.loads(self.cached_request(url=API_ENDPOINT))

        contractIncentives = self.w3.eth.contract(
            address=self.w3.toChecksumAddress(self.incentives_address),
            abi=abi
        )
        return contractIncentives

    @lru_cache
    def get_aToken_contract(self, aTokenAddress):
        """aAAVE token contract"""
        abi = self.config['ethereum']['aAAVE_abi']

        depositAPR_contract = self.w3.eth.contract(
            address=self.w3.toChecksumAddress(aTokenAddress),
            abi=abi
        )
        return depositAPR_contract

    @lru_cache
    def get_vToken_contract(self, variableDebtTokenAddress):
        """variable vToken debt contract"""
        abi = self.config['ethereum']['var_debt_abi']

        borrowAPR_contract = self.w3.eth.contract(
            address=self.w3.toChecksumAddress(variableDebtTokenAddress),
            abi=abi
        )
        return borrowAPR_contract


#########################################################################

class GetRates(Requestbase):
    def __init__(self, chain, store_file):
        self.cmc = CMC
        self.chain = chain
        super(GetRates, self).__init__(chain, store_file)

    def get_token_contract(self, token_address):
        return super().contractv2(token_address)

    def get_token_symbol(self, token_address):
        t = self.get_token_contract(token_address)
        symbol = t.functions.symbol().call()
        return symbol

    def get_token_price(self, token_address):
        t = self.get_token_contract(token_address)
        symbol = t.functions.symbol().call()
        price = self.cmc.get_quote(symbol, 'USD')
        return print('Price of {} is ${}.'.format(symbol, round(price, 2)))

    def get_reserveData(self, token_address):
        """ Returns reserve data from lending contract"""
        contractLending = self.lending_contract()
        reserveData = contractLending.functions.getReserveData(self.w3.toChecksumAddress(token_address)).call()
        return reserveData

    def get_tokenEmissions(self, token_address):
        """ Returns 'a' and 'v' asset EmissionPerSecond"""
        reserveData = self.get_reserveData(token_address)
        aTokenAddress = reserveData[7]
        variableDebtTokenAddress = reserveData[9]

        contractIncentives = self.incentives_contract()
        aEmissionPerSecond = contractIncentives.functions.assets(aTokenAddress).call()[0]
        vEmissionPerSecond = contractIncentives.functions.assets(variableDebtTokenAddress).call()[0]
        return aEmissionPerSecond, vEmissionPerSecond

    def get_apys(self, token_address):
        """Returns any tokens deposit and borrow APY for AAVE and Geist on
        selected chains"""
        RAY = 10 ** 27
        SECONDS_PER_YEAR = 31536000

        reserveData = self.get_reserveData(token_address)
        liquidityRate = reserveData[3]
        variableBorrowRate = reserveData[4]

        depositAPR = liquidityRate / RAY
        variableBorrowAPR = variableBorrowRate / RAY
        stableBorrowAPR = variableBorrowRate / RAY

        depositAPY = ((1 + (depositAPR / SECONDS_PER_YEAR)) ** SECONDS_PER_YEAR) - 1
        variableBorrowAPY = ((1 + (variableBorrowAPR / SECONDS_PER_YEAR)) ** SECONDS_PER_YEAR) - 1
        stableBorrowAPY = ((1 + (stableBorrowAPR / SECONDS_PER_YEAR)) ** SECONDS_PER_YEAR) - 1

        return [round(depositAPY, 4) * 100, round(stableBorrowAPY, 4) * 100]

    def get_rewardAprs(self, token_address):
        """Returns Incentive APR's"""
        WEI_DECIMALS = 10 ** 18
        SECONDS_PER_YEAR = 31536000

        reserveData = self.get_reserveData(token_address)

        aTokenAddress = reserveData[7]
        variableDebtTokenAddress = reserveData[9]

        depositAPR_contract = self.get_aToken_contract(aTokenAddress)
        borrowAPR_contract = self.get_vToken_contract(variableDebtTokenAddress)

        totalATokenSupply = depositAPR_contract.functions.totalSupply().call()
        totalCurrentVariableDebt = borrowAPR_contract.functions.totalSupply().call()

        UNDERLYING_TOKEN_DECIMALS = 10 ** depositAPR_contract.functions.decimals().call()

        aEmissionPerSecond, vEmissionPerSecond = self.get_tokenEmissions(token_address)
        aEmissionPerYear = aEmissionPerSecond * SECONDS_PER_YEAR
        vEmissionPerYear = vEmissionPerSecond * SECONDS_PER_YEAR

        REWARD_PRICE_ETH = self.cmc.get_quote(self.store_file['reward_token'], 'ETH')
        symbol = self.get_token_symbol(token_address).split(".")[0]
        TOKEN_PRICE_ETH = self.cmc.get_quote(symbol, 'ETH')

        incentiveDepositAPRPercent = 100 * (aEmissionPerYear * REWARD_PRICE_ETH * UNDERLYING_TOKEN_DECIMALS) / (
                totalATokenSupply * TOKEN_PRICE_ETH * WEI_DECIMALS)
        incentiveBorrowAPRPercent = 100 * (vEmissionPerYear * REWARD_PRICE_ETH * UNDERLYING_TOKEN_DECIMALS) / (
                totalCurrentVariableDebt * TOKEN_PRICE_ETH * WEI_DECIMALS)
        return [round(incentiveDepositAPRPercent, 2), round(incentiveBorrowAPRPercent, 2)]


if __name__ == '__main__':
    with open(r'abi_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # WAVAX
    token_address = '0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7'
    chain = 'avalanche'

    query = GetRates(chain, config)
    # print(x.get_apys(token_address))
    print(query.get_rewardAprs(token_address))


# # token_address = '0x6B175474E89094C44Da98b954EedeAC495271d0F'
# # chain = 'ethereum'
#
# # chain = 'polygon'
# # token_addresss = '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270'
#
#
# query = GetRates(chain, config)
#
# ## DAI
# # token_address = '0xd586E7F844cEa2F87f50152665BCbc2C279D8d70'
#
# print(query.lending_contract())
