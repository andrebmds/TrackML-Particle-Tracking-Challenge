# Manage input and output for train model 
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import os
import numpy as np


class Preprocessing():
	def __init__(self,X, y):
		self.X = X
		self.y = y

		self.split_data()

	def split_data(self):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
		del self.X, self.y

	

if __name__ == '__main__':
	# np.random.rand(3,2)
	x = np.random.rand(10)
	y = np.random.rand(10)
	# print(x)

	val = Preprocessing(X=x, y=y)
	print(len(val.X_train))
	print(len(val.X_test))