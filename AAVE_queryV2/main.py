import CMC
import yaml
from getRates import GetRates

price = CMC.get_quote('LUNA', 'USD')

with open(r'abi_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

token = 'WAVAX'
token_address = '0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7'
chain = 'avalanche'

# token = 'DAI'
# token_address = '0x6B175474E89094C44Da98b954EedeAC495271d0F'
# chain = 'ethereum'

if __name__ == '__main__':
    price = CMC.get_quote(token, 'USD')
    print('{} price USD: ${}'.format(token, round(price, 2)))
    query = GetRates(chain, config)
    depositAPY, borrowAPY = query.get_apys(token_address)
    incentiveDepositAPR, incentiveBorrowAPR = query.get_rewardAprs(token_address)
    print('\nAAVE V2 rates on {} chain for {}.'.format(chain, token))
    print('Deposit APY: {}%, Borrow APY: {}%'.format(round(depositAPY, 2), borrowAPY))
    print('Deposit Incentive APR: {}%, Borrow Incentive APR: {}%'.format(incentiveDepositAPR, incentiveBorrowAPR))
