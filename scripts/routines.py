__author__ = 'Dante'

import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as acc
import numpy as np
import density_weight as dw
import matplotlib as plt
from matplotlib import pyplot
import math
from sklearn import cross_validation as cv
from collections import Counter
import scipy.cluster.hierarchy as sch
from structures import fingerprinter as fptr
from structures import kernel_fns as kf
from chem import chem


def dw_performance(matrix, classes, filename, initial=2, iterations=100, beta=1.0, C=1.0, gamma=0.005, degree=2, kernel='rbf', batch=1, decf=False, seed=None, simfp=fptr.integer_sim):
	"""This function takes as input the positive and negative samples from a known isozyme and compares how the learning curve
	changes between an SVC trained on an increasing number of randomly selected samples and an SVC trained with an increasing
	number of samples selected via an uncertainty sampling/density weighting method."""

	rand_res = []
	dw_res = []

	k_dict = {'rbf': 'rbf', 'poly': 'poly', 'tanimoto': kf.tanimoto}


	#Iterates n times.
	for iteration in range(iterations):
		#If you set the random_state kwarg, it will make the test set uniform across all iterations.
		x_train, x_test, y_train, y_test = cv.train_test_split(matrix, classes, test_size=0.4, random_state=seed)

		i = range(x_train.shape[0])
		#Change the second argument to start with a different number of samples. The while loop ensures that the initial
		#training smaple is instantiated with at least one example from each class.
		rand_train = random.sample(i, initial)

		while set(y_train[rand_train]) != {-1, 1}:
			del rand_train[-1]
			for index in random.sample(i, 1):
				rand_train.append(index)

		#Initialize the training set for the DW curve with the same points as the random curve.
		dw_train = []
		dw_train += rand_train

		#Results storage.
		n = []
		rand_scores = []
		dw_scores = []

		#Each data point in each iteration si created here.
		while len(rand_train) < x_train.shape[0]:

			clf_rand = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)
			clf_dw = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)

			n.append(len(rand_train))

			#Fit, predict and generate accuracy scores.
			clf_rand.fit(x_train[rand_train], y_train[rand_train])
			clf_dw.fit(x_train[dw_train], y_train[dw_train])

			r = clf_rand.predict(x_test)
			d = clf_dw.predict(x_test)

			rand_scores.append(acc(y_test, r))
			dw_scores.append(acc(y_test, d))

			#Update the available points that can be chosen for random.
			available_rand_ = list(set(i) - set(rand_train))
			if len(available_rand_) != 0 and len(available_rand_) % batch < len(available_rand_):
				for index in random.sample(available_rand_, batch):
					rand_train.append(index)
			elif len(available_rand_) != 0 and len(available_rand_) % batch == len(available_rand_):
				rand_train += available_rand_
			else:
				pass

			#Update the available points that can be chosen for DW, and create index table to maintain identity of each
			#example as they are depleted.
			available_dw_ = list(set(i) - set(dw_train))
			index_table_ = {orig: update for update, orig in enumerate(available_dw_)}
			pairwise_tc_avg = dw.avg_proximity(x_train[available_dw_], x_train[available_dw_], f=simfp)
			if len(available_dw_) != 0 and len(available_dw_) % batch < len(available_dw_):
				if decf:
					xi = [dw.weight(dw.hyper_distance(clf_dw.decision_function(x_train[a])), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				else:
					xi = [dw.weight(dw.entropy(clf_dw.predict_proba(x_train[a])[:, 0]), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				foo = sorted(zip(available_dw_, xi), key=lambda x: x[1], reverse=True)
				dw_train += [ele[0] for ele in foo[:batch]]
			elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
				dw_train += available_dw_
			else:
				pass

		clf_rand_last = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[rand_train], y_train[rand_train])
		clf_dw_last = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[dw_train], y_train[dw_train])
		n.append(len(rand_train))
		r_last = clf_rand_last.predict(x_test)
		d_last = clf_dw_last.predict(x_test)
		rand_scores.append(acc(y_test, r_last))
		dw_scores.append(acc(y_test, d_last))

		rand_res.append(np.array(rand_scores))
		dw_res.append(np.array(dw_scores))


	rand_avg = np.sum(np.vstack(tuple(rand_res)), 0) / iterations
	dw_avg = np.sum(np.vstack(tuple(dw_res)), 0) / iterations
	rand_err = 1.96 * np.std(np.vstack(tuple(rand_res)), 0) / math.sqrt(iterations)
	dw_err = 1.96 * np.std(np.vstack(tuple(dw_res)), 0) / math.sqrt(iterations)
	xdesc = 'Number of Training Samples'
	ydesc = 'Accuracy Score'
	plt.rcParams['font.sans-serif'] = ['Arial']

	pyplot.errorbar(n, rand_avg, fmt='s-', yerr=rand_err, color='darkred', markersize=9, lw=2, label='Random Selection')
	pyplot.errorbar(n, dw_avg, fmt='v-', yerr=dw_err, color='darkblue', markersize=9, lw=2, label='Density Weighted')
	pyplot.tick_params(labelsize=14)
	leg_title = "Final Accuracy = %s\nC = %s" % (str(round(rand_avg[-1], 3) * 100) + '%', str(C))
	pyplot.legend(loc=4, title=leg_title)
	pyplot.xlabel(xdesc, size=18, labelpad=14)
	pyplot.ylabel(ydesc, size=18, labelpad=14)
	pyplot.savefig(filename + "C_%s_decisionf_%s.svg" % (str(C), str(decf)))
	pyplot.show()


def dw_ins(matrix, classes, filename, smiles_acc, initial=2, iterations=100, beta=1.0, C=1.0, gamma=0.005, degree=2, kernel='rbf', decf=False, batch=1, seed=None):
	"""This function trains SVCs--initialized with a number of compounds from randomly partitioned train/test sets--with
	progressively more compounds from the test set selected via active learning and outputs a figure showing the
	distribution of selections of compounds in the unlabelled set."""

	k_dict = {'rbf': 'rbf', 'poly': 'poly', 'tanimoto': kf.tanimoto}

	rankings = {k: [] for k in smiles_acc}

	#Iterates n times.
	for iteration in range(iterations):
		#Set the random_state kwarg, it will make the test set uniform across al iterations.
		x_train, x_test, y_train, y_test = cv.train_test_split(matrix, classes, test_size=0.4, random_state=seed)

		i = range(x_train.shape[0])
		#Change the second argument to start with a different number of samples.
		rand_train = random.sample(i, initial)

		while set(y_train['label'][rand_train]) != {-1, 1}:
			del rand_train[-1]
			for index in random.sample(i, 1):
				rand_train.append(index)

		#Initialize the training set for the DW curve with the same points as the random curve.
		dw_train = []
		dw_train += rand_train

		#Results storage.
		n = []

		#Each data point in each iteration is created here.
		while len(dw_train) < x_train.shape[0]:

			clf_dw = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)

			n.append(len(dw_train))
			#Fit, predict and generate accuracy scores.
			clf_dw.fit(x_train[dw_train], y_train['label'][dw_train])

			#Update the available points that can be chosen for DW, and create index table to maintain identity of each
			#example as they are depleted.
			available_dw_ = list(set(i) - set(dw_train))
			index_table_ = {orig: update for update, orig in enumerate(available_dw_)}
			pairwise_tc_avg = dw.avg_proximity(x_train[available_dw_], x_train[available_dw_])
			if len(available_dw_) != 0 and len(available_dw_) % batch < len(available_dw_):
				#Density weight scores calculated in two lists to find the difference between the score of the "best"
				#compound from the background and the "best" hidden labelled compound.
				if decf:
					xi = [dw.weight(dw.hyper_distance(clf_dw.decision_function(x_train[a])), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				else:
					xi = [dw.weight(dw.entropy(clf_dw.predict_proba(x_train[a])[:, 0]), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				foo = sorted(zip(available_dw_, xi), key=lambda x_: x_[1], reverse=True)

				adds = [ele[0] for ele in foo[:batch]]
				dw_train += adds
				smiles_added = [y_train['smiles'][idx] for idx in adds]
				for s in smiles_added:
					rankings[s].append((len(dw_train) - 2) / batch)

			elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
				smiles_added = [y_train['smiles'][idx] for idx in available_dw_]
				for s in smiles_added:
					rankings[s].append(((len(dw_train) - 2) / batch) + 1)
				dw_train += available_dw_
			else:
				pass

		n.append(len(dw_train))

	final = {k: Counter(v) for k, v in rankings.iteritems()}
	positions = [j + 1 for j in range(max([x for val in rankings.values() for x in val]))]
	data_ = []
	for cpd, ctr in final.iteritems():
		data_.append(np.array([ctr[pos] for pos in positions]))

	data_in_ = np.vstack(tuple(data_)).T

	row = positions
	plt.rcParams['font.sans-serif'] = ['Arial']
	fig = pyplot.figure()
	#Plots a dendrogram above the heatmap.
	axdendro = fig.add_axes([0.06, 0.68, 0.8, 0.27])
	Y = sch.linkage(np.vstack(tuple([fptr.reconstruct_fp(s, fptype='FP2') for s in smiles_acc])), method='single', metric='jaccard')
	Z = sch.dendrogram(Y, orientation='top')
	axdendro.set_xticks([])
	axdendro.set_yticks([])

	#Plotting the heat map.  'pcolor' outputs a mappable object that is used as a mandatory argument to 'colorbar().'
	#add axes arguments are: distance from left, distance from bottom, width, height.

	ax = fig.add_axes([0.06, 0.05, 0.8, 0.6])
	#Grab the order of the leaves of the dendrogram so the heatmap can be reordered to match.
	index = Z['leaves']
	D = data_in_.T[index]
	hmap = ax.pcolor(D.T, cmap='gist_heat')

	horiz = np.arange(data_in_.shape[1]) + 0.5
	vert = np.arange(data_in_.shape[0]) + 0.5

	pyplot.ylim([0, vert[-1] + 0.5])
	pyplot.xlim([0, horiz[-1] + 0.5])
	pyplot.ylabel('Position Selected', size=16)
	ax.set_xticks(horiz, minor=False)
	ax.set_yticks(vert, minor=False)

	names = []
	for s in classes['smiles']:
		name_entry = chem.calc_name(s)
		names.append(unicode(name_entry, "utf-8"))

	col = [names[m] + ' (%s)' % str(classes['label'][m]) for m in index]

	ax.set_xticklabels(col, minor=False, rotation=90, ha='center', size=11)
	ax.set_yticklabels(row, minor=False, size=11)

	#Plots the colorbar on separate axes so that the dendrogram can be aligned to the heatmap alone.
	axcolor = fig.add_axes([0.89, 0.05, 0.02, 0.6])
	cbar = pyplot.colorbar(hmap, cax=axcolor)
	axcolor.set_ylabel('Selection Frequency', size=16, rotation=270)
	#Eliminates white lines in Inkscape due to viewer bug; makes colorbar render with overlapping segments.
	cbar.solids.set_edgecolor("face")

	pyplot.savefig(filename + "C_%s_decisionf_%s.svg" % (str(C), str(decf)))
	pyplot.show()


def dw_exp_val(matrix, classes, filename, excl, initial=2, iterations=100, beta=1.0, C=1.0, gamma=0.005, degree=2, kernel='rbf', batch=1, decf=False, seed=None, simfp=fptr.integer_sim):
	"""This function takes as input the positive and negative samples from a known isozyme plus additional data collected
	experimentally (added in an outside script) using active learning.  A 'random' curve is constructed by splitting the
	whole set, then excluding the newest batch of data from training, and finally calculating the accuracy at incrementally
	increasing training set sizes. To assess 'improvement' due to AL, the 'dw' curve is constructed in the same way
	using training data with an equal number of compounds (to the ones excluded in making the random curve) excluded
	 from training."""

	rand_res = []
	dw_res = []

	k_dict = {'rbf': 'rbf', 'poly': 'poly', 'tanimoto': kf.tanimoto}

	# Iterates n times.
	for iteration in range(iterations):
		#If you set the random_state kwarg, it will make the test set uniform across al iterations.
		x_train, x_test, y_train, y_test = cv.train_test_split(matrix, classes, test_size=0.4, random_state=seed)

		#Exclude experimental compounds from the random curve, and random compounds for the dw curve.
		rand_excl = [i for i, smi in enumerate(y_train['smiles']) if smi in excl]
		dw_excl = random.sample(range(x_train.shape[0]), len(rand_excl))

		i = list(set(range(x_train.shape[0])) - set(rand_excl))
		j = list(set(range(x_train.shape[0])) - set(dw_excl))

		#Change the second argument to start with a different number of samples. The while loop ensures that the initial
		#training sample is instantiated with at least one example from each class.
		rand_train = random.sample(i, initial)
		while set(y_train['label'][rand_train]) != {-1, 1}:
			del rand_train[-1]
			for index in random.sample(i, 1):
				rand_train.append(index)

		#Initialize the training set for the DW curve points randomly chosen from the allowed dw indices.
		dw_train = random.sample(j, initial)
		while set(y_train['label'][dw_train]) != {-1, 1}:
			del dw_train[-1]
			for index in random.sample(j, 1):
				dw_train.append(index)

		#Results storage.
		n = []
		rand_scores = []
		dw_scores = []

		#Each data point in each iteration is created here.
		while len(rand_train) < x_train.shape[0] - len(excl):

			clf_rand = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)
			clf_dw = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)

			n.append(len(rand_train))

			#Fit, predict and generate accuracy scores.
			clf_rand.fit(x_train[rand_train], y_train['label'][rand_train])
			clf_dw.fit(x_train[dw_train], y_train['label'][dw_train])

			r = clf_rand.predict(x_test)
			d = clf_dw.predict(x_test)

			rand_scores.append(acc(y_test['label'], r))
			dw_scores.append(acc(y_test['label'], d))

			#Update the available points that can be chosen for random.
			available_rand_ = list(set(i) - set(rand_train))
			if len(available_rand_) != 0 and len(available_rand_) % batch < len(available_rand_):
				for index in random.sample(available_rand_, batch):
					rand_train.append(index)
			elif len(available_rand_) != 0 and len(available_rand_) % batch == len(available_rand_):
				rand_train += available_rand_
			else:
				pass

			#Update the available points that can be chosen for DW, and create index table to maintain identity of each
			#example as they are depleted.
			available_dw_ = list(set(j) - set(dw_train))
			index_table_ = {orig: update for update, orig in enumerate(available_dw_)}
			pairwise_tc_avg = dw.avg_proximity(x_train[available_dw_], x_train[available_dw_], f=simfp)
			if len(available_dw_) != 0 and len(available_dw_) % batch < len(available_dw_):
				if decf:
					xi = [dw.weight(dw.hyper_distance(clf_dw.decision_function(x_train[a])), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				else:
					xi = [dw.weight(dw.entropy(clf_dw.predict_proba(x_train[a])[:, 0]), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				foo = sorted(zip(available_dw_, xi), key=lambda x: x[1], reverse=True)
				dw_train += [ele[0] for ele in foo[:batch]]
			elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
				dw_train += available_dw_
			else:
				pass

		clf_rand_last = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[rand_train], y_train['label'][rand_train])
		clf_dw_last = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[dw_train], y_train['label'][dw_train])
		n.append(len(rand_train))
		r_last = clf_rand_last.predict(x_test)
		d_last = clf_dw_last.predict(x_test)
		rand_scores.append(acc(y_test['label'], r_last))
		dw_scores.append(acc(y_test['label'], d_last))

		rand_res.append(np.array(rand_scores))
		dw_res.append(np.array(dw_scores))

	rand_avg = np.sum(np.vstack(tuple(rand_res)), 0) / iterations
	dw_avg = np.sum(np.vstack(tuple(dw_res)), 0) / iterations
	rand_err = 1.96 * np.std(np.vstack(tuple(rand_res)), 0) / math.sqrt(iterations)
	dw_err = 1.96 * np.std(np.vstack(tuple(dw_res)), 0) / math.sqrt(iterations)
	xdesc = 'Number of Training Samples'
	ydesc = 'Accuracy Score'
	plt.rcParams['font.sans-serif'] = ['Arial']

	pyplot.errorbar(n, rand_avg, fmt='s-', yerr=rand_err, color='darkred', markersize=9, lw=2,
					label='Random Selection')
	pyplot.errorbar(n, dw_avg, fmt='v-', yerr=dw_err, color='darkblue', markersize=9, lw=2,
	                label='Density Weighted')
	pyplot.tick_params(labelsize=14)
	leg_title = "Final Random Accuracy = %s\nFinal DW Accuracy = %s\nC = %s" % (str(round(rand_avg[-1], 3) * 100) + '%', str(round(dw_avg[-1], 3) * 100) + '%', str(C))
	pyplot.legend(loc=4, title=leg_title)
	pyplot.xlabel(xdesc, size=18, labelpad=14)
	pyplot.ylabel(ydesc, size=18, labelpad=14)
	pyplot.savefig(filename + "C_%s_decisionf_%s.svg" % (str(C), str(decf)))
	pyplot.show()

def dw_exp_ins(matrix, classes, filename, smiles_acc, excl, initial=2, iterations=100, beta=1.0, C=1.0, gamma=0.005, degree=2, kernel='rbf', batch=1, decf=False, seed=None, simfp=fptr.integer_sim):
	"""This function takes as input the positive and negative samples from a known isozyme plus additional data collected
	experimentally (added in an outside script) using active learning.  A 'random' curve is constructed by splitting the
	whole set, then excluding the newest batch of data from training, and finally calculating the accuracy at incrementally
	increasing training set sizes. To assess 'improvement' due to AL, the 'dw' curve is constructed in the same way
	using training data with an equal number of compounds (to the ones excluded in making the random curve) excluded
	from training."""

	rand_res = []
	dw_res = []

	rankings = {k: [] for k in smiles_acc}

	k_dict = {'rbf': 'rbf', 'poly': 'poly', 'tanimoto': kf.tanimoto}

	# Iterates n times.
	for iteration in range(iterations):
		#If you set the random_state kwarg, it will make the test set uniform across al iterations.
		x_train, x_test, y_train, y_test = cv.train_test_split(matrix, classes, test_size=0.4, random_state=seed)

		#Exclude experimental compounds from the random curve, and random compounds for the dw curve.
		rand_excl = [i for i, smi in enumerate(y_train['smiles']) if smi in excl]
		dw_excl = random.sample(range(x_train.shape[0]), len(rand_excl))

		j = list(set(range(x_train.shape[0])) - set(dw_excl))

		#Initialize the training set for the DW curve points randomly chosen from the allowed dw indices.
		dw_train = random.sample(j, initial)
		while set(y_train['label'][dw_train]) != {-1, 1}:
			del dw_train[-1]
			for index in random.sample(j, 1):
				dw_train.append(index)

		#Results storage.
		n = []

		#Each data point in each iteration is created here.
		while len(dw_train) < x_train.shape[0] - len(excl):

			clf_dw = SVC(C=C, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)

			n.append(len(dw_train))

			#Fit, predict and generate accuracy scores.
			clf_dw.fit(x_train[dw_train], y_train['label'][dw_train])

			#Update the available points that can be chosen for DW, and create index table to maintain identity of each
			#example as they are depleted.
			available_dw_ = list(set(j) - set(dw_train))
			index_table_ = {orig: update for update, orig in enumerate(available_dw_)}
			pairwise_tc_avg = dw.avg_proximity(x_train[available_dw_], x_train[available_dw_], f=simfp)
			if len(available_dw_) != 0 and len(available_dw_) % batch < len(available_dw_):
				if decf:
					xi = [dw.weight(dw.hyper_distance(clf_dw.decision_function(x_train[a])), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				else:
					xi = [dw.weight(dw.entropy(clf_dw.predict_proba(x_train[a])[:, 0]), pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
				foo = sorted(zip(available_dw_, xi), key=lambda x: x[1], reverse=True)

				adds = [ele[0] for ele in foo[:batch]]
				dw_train += adds
				smiles_added = [y_train['smiles'][idx] for idx in adds]
				for s in smiles_added:
					rankings[s].append((len(dw_train) - 2) / batch)

			elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
				smiles_added = [y_train['smiles'][idx] for idx in available_dw_]
				for s in smiles_added:
					rankings[s].append(((len(dw_train) - 2) / batch) + 1)
				dw_train += available_dw_
			else:
				pass

		n.append(len(dw_train))
		smiles_added = [y_train['smiles'][idx] for idx in available_dw_]
		for s in smiles_added:
			rankings[s].append(((len(dw_train) - 2) / batch) + 1)

	final = {k: Counter(v) for k, v in rankings.iteritems()}
	positions = [j + 1 for j in range(max([x for val in rankings.values() for x in val]))]
	data_ = []
	for cpd, ctr in final.iteritems():
		data_.append(np.array([ctr[pos] for pos in positions]))

	data_in_ = np.vstack(tuple(data_)).T

	row = positions
	plt.rcParams['font.sans-serif'] = ['Arial']
	fig = pyplot.figure()
	# Plots a dendrogram above the heatmap.
	axdendro = fig.add_axes([0.06, 0.68, 0.8, 0.27])
	Y = sch.linkage(np.vstack(tuple([fptr.reconstruct_fp(s, fptype='FP2') for s in smiles_acc])), method='single',
	                metric='jaccard')
	Z = sch.dendrogram(Y, orientation='top')
	axdendro.set_xticks([])
	axdendro.set_yticks([])

	#Plotting the heat map.  'pcolor' outputs a mappable object that is used as a mandatory argument to 'colorbar().'
	#add axes arguments are: distance from left, distance from bottom, width, height.

	ax = fig.add_axes([0.06, 0.05, 0.8, 0.6])
	#Grab the order of the leaves of the dendrogram so the heatmap can be reordered to match.
	index = Z['leaves']
	D = data_in_.T[index]
	hmap = ax.pcolor(D.T, cmap='gist_heat')

	horiz = np.arange(data_in_.shape[1]) + 0.5
	vert = np.arange(data_in_.shape[0]) + 0.5

	pyplot.ylim([0, vert[-1] + 0.5])
	pyplot.xlim([0, horiz[-1] + 0.5])
	pyplot.ylabel('Position Selected', size=16)
	ax.set_xticks(horiz, minor=False)
	ax.set_yticks(vert, minor=False)

	names = []
	for s in classes['smiles']:
		name_entry = chem.calc_name(s)
		names.append(unicode(name_entry, "utf-8"))

	col = [names[m] + ' (%s)' % str(classes['label'][m]) for m in index]

	ax.set_xticklabels(col, minor=False, rotation=90, ha='center', size=11)
	ax.set_yticklabels(row, minor=False, size=11)

	#Plots the colorbar on separate axes so that the dendrogram can be aligned to the heatmap alone.
	axcolor = fig.add_axes([0.89, 0.05, 0.02, 0.6])
	cbar = pyplot.colorbar(hmap, cax=axcolor)
	axcolor.set_ylabel('Selection Frequency', size=16, rotation=270)
	#Eliminates white lines in Inkscape due to viewer bug; makes colorbar render with overlapping segments.
	cbar.solids.set_edgecolor("face")

	pyplot.savefig(filename + "C_%s_decisionf_%s.svg" % (str(C), str(decf)))
	pyplot.show()


