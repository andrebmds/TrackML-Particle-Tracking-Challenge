import walletDB

# Amount | Coin to Coin | Convert 
# 0.5	 | BTC to BRL	| 35000

# Objective 
# Grow amount in each coin 

# Rule for buy and sell 
	# -1 sell
	#  0 nothin
	#  1 buy
# amount



# workflow:
	# Input
		# Read orderbook
		# Read wallet 

	# make prediction
		# Model for prediction :
			# CNN
			# KNN
			# Garsh
			# etc.
	# Output
		# Buy
	 	# Sell
	  	# Nothing

	# Check improvement for each action 
	 	# By verifying the grow for each coin

class output():
	def __init__(self, wallet):
		self.wallet = wallet

	def buy(self):
		# RemoveAmount
		# Add Amount
		pass

	def sell(self):
		# RemoveAmount
		# Add Amount
		pass

	def nothing(self):
		# Do nothing, waiting for better moment.
		pass

class input():
	def __init__(self):
		pass

	def readWallet():
		pass

if __name__ == '__main__':
# Read wallet 
	wallet = walletDB.Wallet(name_of_wallet='walletQLearning.csv')
	# Make decision
	output(wallet)