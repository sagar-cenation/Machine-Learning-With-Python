import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings # warn user when using dumb number for k
from collections import Counter
#dont forget this
import pandas as pd
import random # to shuffle the dataset
style.use('fivethirtyeight')


# for i in dataset:
# 	for ii in dataset[i]:
# 		plt.scatter(ii[0], ii[1], s=100, color=i)
# same as below single line
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')
	distances = []
	for group in data:
		for features in data[group]:
			# euclidean_distance = np.sqrt( np.sum((np.array(features)-np.array(predict))**2))
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) # faster
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	# print(vote_result,confidence)
	return vote_result,confidence

	# votes = [i[1] for i in sorted(distances)[:k]]
 #    vote_result = Counter(votes).most_common(1)[0][0]
 #    return vote_result

accuracies = []
for i in range(25):
	# shuffle the data`
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)
	# print(df.head())
	full_data = df.astype(float).values.tolist() 
	# print(full_data[:5])
	random.shuffle(full_data)

	# slice the data
	test_size = 0.4
	train_set = {2:[],4:[]}
	test_set = {2:[],4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])

	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			vote,confidence = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct += 1
			# else:
				# print(confidence)
			total += 1

	# print('Accuracy:', float(correct)/float(total))
	accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))