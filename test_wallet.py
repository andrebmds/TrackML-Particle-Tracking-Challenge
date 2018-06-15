import unittest
import walletDB
import os

class WalletTestCase(unittest.TestCase):
	def setUp(self):
		""" setUp is called before every test"""
		# To make the tests remove file test initial to 
		# not interfiring whit tests
		try:
			os.remove(os.path.join('data','wallet_test.csv'))
		except:
			pass

	def tearDown(self):
		'''tearDown is called at the end of every test'''
		pass

	def testWalletNumberOfElementsAdd(self):
		'''All methods beginning with 'test' are executed'''
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 10, coin = 'BTC')
		
		self.assertEqual(wallet_main.wallet['coin'].count(), 1, 'Number of row should be 1')

	def testCheckColumnsName(self):
		# self.assertEqual(self.wallet.wallet.columns.values, ['date', 'coin', 'amount'])
		pass

	def testCheckTotalAmountIsEqualTo10(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 10, coin = 'BTC')
		total = wallet_main.getTotalAmount('BTC')

		self.assertEqual(total, 10, 'Is\'n {} return correct value'.format(total))

	def testAmountDiferentsAmountsForBTC(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 10, coin = 'BTC')
		wallet_main.make_deposit(amount = -5, coin = 'BTC')
		total = wallet_main.getTotalAmount('BTC')

		self.assertEqual(total, 5, 'Is\'n {} return correct value'.format(total))

	def testAmountDiferentsAmountsForLTC(self):
		coin = 'LTC'
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 10, coin = coin)
		wallet_main.make_deposit(amount = -5, coin = coin)
		total = wallet_main.getTotalAmount(coin)

		self.assertEqual(total, 5, 'Is\'n {} return correct value'.format(total))

	def testAmountDiferentsReultIsZero(self):
		coin = 'LTC'
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 0.5, coin = coin)
		wallet_main.make_deposit(amount = -0.5, coin = coin)
		total = wallet_main.getTotalAmount(coin)

		self.assertEqual(total, 0, 'Is\'n {} return correct value'.format(total))

	def testAmountDiferentsCoins(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 0.5, coin = 'BTC')
		wallet_main.make_deposit(amount = -5, coin = 'LTC')
		total = wallet_main.getTotalAmount('BTC')

		self.assertEqual(total, 0.5, 'Is\'n {} return correct value'.format(total))

	def testOpenAndCloseSeveralTimes(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 5, coin = 'BTC')
		del wallet_main

		wallet_main2 = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main2.make_deposit(amount = 5, coin = 'BTC')
		del wallet_main2

		wallet_main3 = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main3.make_deposit(amount = 5, coin = 'BTC')
		total = wallet_main3.getTotalAmount('BTC')
		
		self.assertEqual(total, 15, 'Is\'n {} return correct value'.format(total))

	def testListOfCoins(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 5, coin = 'BTC')
		wallet_main.make_deposit(amount = 5, coin = 'LTC')
		
		self.assertEqual(set(wallet_main.showCoins()), set(['BTC','LTC']), 'Output of cois is different of expept')

	def testReadMultipleCoins(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 5, coin = 'BTC')
		wallet_main.make_deposit(amount = 25, coin = 'BTC')
		wallet_main.make_deposit(amount = 5, coin = 'LTC')

		self.assertEqual(set([wallet_main.getTotalAmount(coin) for coin in ['BTC','LTC']]),set([30, 5]))

	def testShowHistory(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 5, coin = 'BTC')
		wallet_main.make_deposit(amount = 25, coin = 'BTC')
		wallet_main.make_deposit(amount = 5, coin = 'LTC')
		first = wallet_main.showHistoryAmount('BTC')['totalAmount'].iloc[0]
		second = wallet_main.showHistoryAmount('BTC')['totalAmount'].iloc[1]
		self.assertEqual(first, 5)
		self.assertEqual(second, 30)
	
	def testNormalizeHistory(self):
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = 5, coin = 'BTC')
		wallet_main.make_deposit(amount = 25, coin = 'BTC')
		wallet_main.make_deposit(amount = 5, coin = 'LTC')

		normalize = wallet_main.normalize('amount')
		

if __name__ == '__main__':
	# run all TestCase's in this module
	unittest.main()