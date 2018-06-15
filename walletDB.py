import numpy as np 
import pandas as pd 
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


'''Provide a history of all moviments of my wallet'''

# NAME # DATE      | COIN 	| AMOUNT
# TYPE # UNIX TIME | STR 	| FLOAT

class Wallet():
	wallet_dir = 'data'
	colors = ['#E97778','#89C7B6','#FFD57E','#AD84C7','#7998C9']

	def __init__(self,name_of_wallet='wallet.csv'):

		self.wallet_path = os.path.join(self.wallet_dir,
										name_of_wallet)

		if not os.path.exists(self.wallet_dir):
			os.makedirs(self.wallet_dir)
		if os.path.exists(self.wallet_path):
			self.loadHistory()
		else:
			wallet_columns = ['date','coin', 'amount']
			self.wallet = pd.DataFrame(columns = wallet_columns)
			self.saveHistory()

	def make_deposit(self, date = time.time(), amount = 0,
					coin = 'BTC'):

		new_deposit = pd.DataFrame([{'date': date, 'coin':coin,
									'amount':amount}])
		self.wallet = self.wallet.append(new_deposit)
		self.saveHistory()

	def getHistoryFromCoin(self, coin):
		return self.wallet[self.wallet['coin'] == coin]

	def getTotalAmount(self, coin):
		coin_df = self.getHistoryFromCoin(coin)
		return coin_df['amount'].sum()

	def loadHistory(self):
		self.wallet = pd.read_csv(self.wallet_path)

	def saveHistory(self):
		self.wallet.to_csv(self.wallet_path, index=False)

	def showCoins(self):
		return self.wallet.coin.unique()

	def showHistoryAmount(self, coin):
		coin = self.getHistoryFromCoin(coin)
		coin['totalAmount'] = coin['amount'].cumsum()
		return coin

	def plotBalance(self):
		coins = self.showCoins()
		sizes = [self.getTotalAmount(coin) for coin in coins]
		plt.pie(sizes, labels= coins, autopct='%1.1f%%',
				shadow=False, colors = self.colors)
		plt.axes().set_aspect('equal','datalim')
		plt.show()

	def normalize(self, df, key):
		normalize = df[key]/df[key].max()
		return normalize

	def plotHistory(self):
		coins = self.showCoins()
		coinHistory = [self.showHistoryAmount(coin) for coin in coins]
		[plt.plot(coin['date'], coin['totalAmount']) for coin in coinHistory]
		plt.show()


if __name__ == '__main__':
	wallet = Wallet()
	# wallet.make_deposit(amount = 100, coin = 'BRL')
	wallet.plotHistory()