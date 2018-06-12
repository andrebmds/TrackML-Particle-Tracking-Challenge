####################################
# Main purpose is to manage the data
####################################

import os
import SetModel


import numpy as np
import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from trackml.score import score_event
from trackml.dataset import load_event, load_dataset

from sklearn.preprocessing import StandardScaler


# PATH FILES FOR INPUT
# path_to_train = 'input/train_100_events'
path_to_train = 'input/train_3'
list_of_events = os.listdir(path_to_train)
print('Len path_to_train: ',len(list_of_events))
print('Name:', list_of_events[0])

# CORRECT NAMES
cut_list = np.array([list_of_events[i][0:14] for i in range(len(list_of_events))])
cut_list = np.unique(cut_list)
print('Len cut_list: ',len(cut_list))
print(cut_list[0])


# SAMPLE FOR TRAINING 
def get_training_sample(path_to_data, event_names):

	events = []
	track_id = 0

	for name in tqdm(event_names):
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
	# Put all hits into one sample with unique track ids
	data = pd.concat(events, axis=0)

	return data

def _preprocess(hits):
		
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


def create_one_event_submission(event_id, hits, labels):
	sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
	submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
	return submission

train_data = get_training_sample(path_to_train, cut_list)
print(train_data.head())
print(train_data.info())

X = _preprocess(train_data)
y = train_data.particle_id.values

print(type(X))
print('Len: ', len(X))
print('Shape: ', X.shape[1])
print('dim: ', 	 X.ndim)
print('Size: ',  X.size)

# PLOT SCATTER 3D
# fig = pyplot.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:,0],X[:,1],X[:,2])
# pyplot.show()

model = SetModel.SetUpModel(X= X, y= y, batch=50)

# MAKE PREDICTION
# READ FILE FOR SUBMITION
# X_sub = 
hits, cells, particles, truth = load_event(os.path.join(path_to_train, cut_list[11]))

X_predict = _preprocess(X)
labels = model.predict(X_predict)

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)





