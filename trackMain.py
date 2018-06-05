import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


from sklearn.preprocessing import MultiLabelBinarizer
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from subprocess import check_output

# Change this according to your directory preferred setting
path_to_train = 'input/train_100_events'
list_of_events = os.listdir(path_to_train)
print('Len path_to_train: ',len(list_of_events))
print('Name:', list_of_events[0])

# Get correct names
cut_list = np.array([list_of_events[i][0:14] for i in range(len(list_of_events))])
cut_list = np.unique(cut_list)
print('Len cut_list: ',len(cut_list))
print(cut_list[0])
	

def get_training_sample(path_to_data, event_names):

	events = []
	track_id = 0

	for name in event_names:
		print(name)
		# Read an event
		hits, cells, particles, truth = load_event(os.path.join(path_to_data, name))

		# Generate new vector of particle id
		particle_ids = truth.particle_id.values
		particle2track = {}
		for pid in np.unique(particle_ids):
			particle2track[pid] = track_id
			track_id += 1
		hits['particle_id'] = [particle2track[pid] for pid in particle_ids]

		# Collect hits
		events.append(hits)
	# Put all hits into one sample with unique tracj ids
	data = pd.concat(events, axis=0)

	return data

train_data = get_training_sample(path_to_train, cut_list)

print(train_data.info())

class Clusterer(object):
	
	def __init__(self):
		self.classifier = None
	
	def _preprocess(self, hits):
		
		x = hits.x.values
		y = hits.y.values
		z = hits.z.values

		r = np.sqrt(x**2 + y**2 + z**2)
		hits['x2'] = x/r
		hits['y2'] = y/r
		hits['z2'] = z/r

		ss = StandardScaler()
		X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
		
		return X

	def model(self):
		model = Sequential()
		model.add(Dense(12,
					input_dim=3,
					kernel_initializer='uniform',
					activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
		
		# model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))  model.add(Activation('relu'))
		# model.add(Conv2D(32,(3,3)))
		# model.add(Activation('relu'))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(Dropout(0.25))
		# model.add(Conv2D(64,(3,3), padding='same'))
		# model.add(Activation('relu'))
		# model.add(Conv2D(64,(3, 3)))
		# model.add(Activation('relu'))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(Dropout(0.25))
		# model.add(Flatten())
		# model.add(Dense(512))
		# model.add(Activation('relu'))
		# model.add(Dropout(0.5))
		# model.add(Dense(num_classes))
		# model.add(Activation('softmax'))
		
		
		optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

		model.compile(optimizer = optimizer , loss = "mse", metrics=["accuracy"])
		# Save model
		# plot_model(model, to_file='model.png')
		# SVG(model_to_dot(model).create(prog='dot', format='svg'))

		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)

		self.classifier = model
	
	
	def fit(self, hits):
		X = self._preprocess(hits)
		y = hits.particle_id.values
		
		# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
		
		# print('X shape[0]:', X_train.shape[0])
		# print('X len: ', len(X))
		# print('X dim:', X.ndim)
		# print('')
		self.classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
		self.classifier.fit(X, y)

		# self.model()
		# checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
		# self.classifier.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val), callbacks = [checkpointer])
		
	
	def predict(self, hits):
		
		X = self._preprocess(hits)
		labels = self.classifier.predict(X)
		
	    # predictions = self.classifier.predict(X)
	    # print(type(predictions))
	     # np.save('predictions',predictions)
	    # predictions.save('predictions.npz')
	
		return labels

model = Clusterer()
model.fit(train_data)

hits, cells, particles, truth = load_event(os.path.join(path_to_train, cut_list[11]))
labels = model.predict(hits)
print(labels)


def create_one_event_submission(event_id, hits, labels):
	sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
	submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
	return submission

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)

print("Your score: ", score)

load_dataset(path_to_train, skip=1000, nevents=5)

dataset_submissions = []
dataset_scores = []

for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=1000, nevents=5):
		
	# Track pattern recognition
	labels = model.predict(hits)
		
	# Prepare submission for an event
	one_submission = create_one_event_submission(event_id, hits, labels)
	dataset_submissions.append(one_submission)
	
	# Score for the event
	score = score_event(truth, one_submission)
	dataset_scores.append(score)
	
	print("Score for event %d: %.3f" % (event_id, score))
	
print('Mean score: %.3f' % (np.mean(dataset_scores)))

path_to_test = "input/test"
test_dataset_submissions = []

create_submission = True # True for submission 

if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submission = pd.concat(test_dataset_submissions, axis=0)
    submission.to_csv('submission.csv.gz', index=False, compression='gzip')
