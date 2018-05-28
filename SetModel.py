from keras.models import Sequential
from keras.layers import Activation,
						 Dense,
						 Dropout, 
						 Conv2D, 
						 MaxPooling2D, 
						 Flatten, 
						 Embedding, 
						 LSTM

# Chosse a model 
class Generate_Models():
	def __init__(self):
		self.model = None

	def mlp_binary_classification(self, input_dim = 3, output_dim = 1):
		model = Sequential()

		model.add(Dense(12,
						input_dim = input_dim,
						kernel_initializer='uniform',
						activation='relu'))
		model.add(Dense(8, kernel_initializer='uniform',activation='relu'))
		model.add(Dense(output_dim, kernel_initializer='uniform',activation='sigmoid'))

		self.check_model(model)
		return model

	def mlp_multi_class_classification(self,input_shape=(784,), output_dim=10):
		model = Sequential()

		model.add(Dense(512, activation='relu',input_shape=input_shape))
		model.add(Dropout(0.2))
		model.add(Dense(512, activation='relu'))
		model.add(Dense(output_dim, activation='softmax'))

		self.check_model(model)
		return model

	def regression(self, input_dim=3):
		model = Sequential()

		model.add(Dense(64, activation='relu', input_dim=input_dim))
		model.add(Dense(1))

		self.check_model(model)
		return model

	def convolutional_neural_network(self, input_shape = (10,), num_classes=10):
		model = Sequential()

		model.add(Conv2D(32,(3,3), padding='same', input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(Conv2D(32,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64,(3,3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes))
		model.add(Activation('softmax'))

		self.check_model(model)
		return model

	def recurrent_neural_network(self):
		model = Sequential()

		model.add(Embedding(20000,128))
		model.add(LSTM(129, dropout=0.2, recurrent_dropout=0.2))
		model.add(Dense(1, activation='sigmoid'))

		self.check_model(model)
		return model

	def check_model(self, model):
		print('------check_model------')
		print('nnOutput shape: ', model.output_shape)
		print(model.summary())
		# print(model.get_config())
		# print(model.get_weights())

class SetUpModel():
	def __init__(self):
		self.model = Sequential()
		pass

	def preprocessing(self):
		pass

	def model_architecture(self):
		pass

	def compile(self):
		pass

	def training(self):
		pass

	def evaluate(self):
		pass

	def prediction(self):
		pass




if __name__ == '__main__':
	model = Generate_Models()
	# model.mlp_binary_classification(input_dim=3, output_dim=1)
	# model.mlp_multi_class_classification(input_shape=(10,), output_dim=1)
	# model.regression(inputdim=3)
	# model.convolutional_neural_network(input_shape=(64,64,3), num_classes=10)
	model.recurrent_neural_network()
