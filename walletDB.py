import numpy as np 
import pandas as pd 
import datetime
import os

class Wallet():
	wallet_dir = 'data'

	def __init__(self,name_of_wallet='wallet.csv'):

		self.wallet_path = os.path.join(self.wallet_dir, name_of_wallet)

		if not os.path.exists(self.wallet_dir):
			os.makedirs(self.wallet_dir)
		if os.path.exists(self.wallet_path):
			self.loadHistory()
		else:
			wallet_columns = ['date', 'coin', 'amount']
			self.wallet = pd.DataFrame(columns = wallet_columns)
			self.saveHistory()

	def make_deposit(self, date = datetime.datetime.today(), amount = 0, coin = 'BTC'):
		new_deposit = pd.DataFrame([{'date': date, 'coin':coin, 'amount':amount}])
		self.wallet = self.wallet.append(new_deposit)
		self.saveHistory()

	def getTotalAmount(self, coin):
		coin_df = self.wallet[self.wallet['coin'] == coin]
		return coin_df['amount'].sum()

	def loadHistory(self):
		self.wallet = pd.read_csv(self.wallet_path)

	def saveHistory(self):
		self.wallet.to_csv(self.wallet_path)


if __name__ == '__main__':
	wallet = Wallet()
	wallet.make_deposit(amount = 10, coin = 'BTC')
	wallet.make_deposit(amount = 5,  coin = 'BTC')
	wallet.make_deposit(amount = 12, coin = 'LTC')

	print(wallet.getTotalAmount('BTC'))
