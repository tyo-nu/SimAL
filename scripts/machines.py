__author__ = 'Dante'

from sklearn.svm import SVC
import numpy as np
import copy
from sklearn.metrics.pairwise import rbf_kernel as radial
import os
from sys import platform as _platform
from structures.isozyme import BrendaIsozyme as bi
import json
from utils.moldraw import MolGrid
import itertools


CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")
if _platform == 'win32':
	DESTPATH = 'C:\\Users\\Dante\\Documents\\MachineLearningProject\\MenDOCS'
else:
	DESTPATH = os.getcwd()

EXC = [296, 1, 2, 3, 4, 297, 298, 299, 308, 312, 313, 307, 295]


def _combine_samples(x_pos, x_neg, labels=None):
	"""Combines sample data in one large numpy array and labels them according to class."""

	y = []

	y += [1] * x_pos.shape[0]
	y += [-1] * x_neg.shape[0]

	x = np.vstack((x_pos, x_neg))

	return x, np.array(y)


def svm_clf(x_pos, x_neg, x_test, C=5, kernel='rbf', gamma=0.005, ent=True, degree=3):
	#'x' is a list of tuples of the form (smiles, fp_array). This is drawn from the
	#rxt_groups attribute of the Isozyme class. 'test' should be in the same format.

	x_pos_array = np.vstack(tuple([t[1] for t in x_pos]))
	x_neg_array = np.vstack(tuple([t[1] for t in x_neg]))
	test_array = np.vstack(tuple([t[1] for t in x_test if t[1] is not None and len(t[1]) == 313]))

	x_array, obj = _combine_samples(x_pos_array, x_neg_array)

	clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=True)
	clf.fit(x_array, obj)

	if ent:
		#It may be necessary to do several probability predictions since the initial Platt scaling can change across
		#run of the same compounds.  If the sample size is small, a loop maknig several classifiers will be
		#necessary.
		labels = list(clf.predict(test_array))

		return zip(labels, list(clf.predict_proba(test_array)[:, 0]))

	else:

		return zip(list(clf.predict(test_array)), list(clf.decision_function(test_array)))

def svm_feature_rank(x_pos, x_neg, C=5, kernel='rbf', gamma=0.005, draw=None):

	with open(os.path.join(CHEMPATH, 'SMARTSFileFull.json'), 'r') as f:
		patts = {int(k.encode('utf-8')): v.split(': ')[1].encode('utf-8') for k, v in json.load(f).iteritems()}

	x_pos_array = np.vstack(tuple([t[1] for t in x_pos]))
	x_neg_array = np.vstack(tuple([t[1] for t in x_neg]))

	smiles_ = [t[0] for t in x_pos] + [t[0] for t in x_neg]

	x_array, obj = _combine_samples(x_pos_array, x_neg_array)

	clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
	clf.fit(x_array, obj)

	gram = radial(clf.support_vectors_, clf.support_vectors_, gamma=gamma)
	alpha = clf.dual_coef_

	objective = 0.5 * float(np.dot(np.dot(alpha, gram), alpha.T))

	responses = []

	features = [i for i in range(x_array.shape[1]) if i + 1 not in EXC]
	#comb_ = itertools.combinations_with_replacement(features, 2)

	for c in features:
		d = copy.deepcopy(clf.support_vectors_)
		#a_, b_ = c[0], c[1]

		d[:, c] = np.zeros(d.shape[0])
		#d[:, b_] = np.zeros(d.shape[0])

		gram_prime = radial(d, d, gamma=gamma)
		objective_prime = 0.5 * float(np.dot(np.dot(alpha, gram_prime), alpha.T))

		sensitivity = (objective - objective_prime) / (objective)

		#responses.append(((a_ + 1, b_ + 1), round(sensitivity, 3)))
		responses.append((c + 1, round(sensitivity, 3)))

	ranks = sorted(responses, key=lambda x: x[1], reverse=True)

	if draw is not None:

		smarts_q = [(qq[0], patts[qq[0]].strip('\n')) for qq in ranks[:draw]]

		mg = MolGrid()
		for smi_ in smiles_:
			mg.add_smiles(smi_)
		fname = 'MolsGrid%s.svg' % '_'.join([str(sm[0]) for sm in smarts_q])
		mg.generate_figures(fname, highlight=tuple([sm[1] for sm in smarts_q]))

	return ranks

if __name__ == "__main__":
	a = bi("Escherichi coli K12", "2.2.1.9")
	a.add_from_sdf('MenD_rd1_good.sdf', 0, pos=True)
	a.add_from_sdf('MenDNegatives.sdf', 0, pos=False)
	ranks = svm_feature_rank(a.pos[0], a.neg[0], C=5)
	f = open('feat_importance.txt', 'w')
	for r in ranks:
		f.write(str(r) + '\n')
	f.close()