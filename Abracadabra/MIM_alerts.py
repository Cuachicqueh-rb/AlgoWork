from web3 import Web3
import requests
import smtplib
from email.message import EmailMessage
import json
import time

def email_alert(subject, body, to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = ## insert email address
    msg['from'] = user
    password = ## insert password

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)

    server.quit()

def MIM_Replenish(config):
    token, contract_address, previous_base = config

    w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/ae3b44ea264a4fdf8060b4be35289905'))
    url_eth = "https://api.etherscan.io/api?module=contract&action=getabi&address="

    API_ENDPOINT = '{}{}'.format(url_eth, contract_address)
    r = requests.get(url=API_ENDPOINT)
    response = r.json()

    contract = w3.eth.contract(
        address=Web3.toChecksumAddress(contract_address),
        abi=response["result"]
    )

    elastic, base = contract.functions.totalBorrow().call()
    elastic = float(Web3.fromWei(elastic, 'ether'))
    base = float(Web3.fromWei(base, 'ether'))
    new_balance = base - float(previous_base)

    if new_balance > 1000000:
        subject = "{}{}".format('MIM Supply Alert for ', token)
        body = "MIM Borrow supply of {} has been added for {}.".format(new_balance, token)
        to = #email address
        email_alert(subject, body, to)
        print('New supply has been added for', token + str(','), 'and alert has been sent.')

        # Set the new base to the config file

        with open('config_abra.json', 'r') as f:
            config = json.load(f)

        config[token].remove(config[token][-1])
        config[token].append(base)

        with open('config_abra.json', 'w') as f:
            json.dump(config, f)
    else:
        print('Base has not changed for', token + str('.'), 'No alert sent.')

if __name__ == '__main__':
    while True:
        with open('config_abra.json', 'r') as f:
            config = json.load(f)
        for i in config.keys():
            MIM_Replenish(config[i])
            time.sleep(5)
        print()
        time.sleep(300)
