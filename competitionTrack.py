import os

import numpy as np
import pandas as pd  
from trackml.dataset import load_event, load_dataset


def get_names_from(file_name='input/train_100_events'):

	# print(list_of_events)

	events_names = np.array([list_of_events[i][0:14] for i in range(len(list_of_events))])
	events_names = np.unique(events_names)
	# print(events_names)
	return events_names

def get_training_sample(path_to_data, event_names):

	events = []
	track_id = 0

	for name in event_names:

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


if __name__ == '__main__':
	# Chack path 
	main_path = 'input'
	train_sample = 'train_100_events'
	path_of_events = os.path.join(main_path, train_sample)
	list_of_events = os.listdir(path_of_events)

	events_names = get_names_from(file_name=path_of_events)

	data = get_training_sample(path_of_events,events_names)
	print(data.head())
	