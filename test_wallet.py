import unittest
import walletDB
import os

class WalletTestCase(unittest.TestCase):
	def setUp(self):
		""" setUp is called before every test"""
		# self.wallet_main = walletDB.Wallet()
		# To make the tests remove file test initial to 
		# not interfiring whit tests
		try:
			os.remove(os.path.join('data','wallet_test.csv'))
			# print('Remove file')
		except:
			pass
		# 	print('File don\'t exist: ')

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

	def testOpenAndCloseSeveralTimes():
		wallet_main = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main.make_deposit(amount = -5, coin = 'BTC')
		wallet_main2 = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main2.make_deposit(amount = -5, coin = 'BTC')
		wallet_main3 = walletDB.Wallet(name_of_wallet='wallet_test.csv')
		wallet_main3.make_deposit(amount = -5, coin = 'BTC')

		total = wallet_main3.getTotalAmount('BTC')
		self.assertEqual(total, -15, 'Is\'n {} return correct value'.format(total))


if __name__ == '__main__':
	# run all TestCase's in this module
	unittest.main()