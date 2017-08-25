__author__ = 'Dante'

import numpy as np


def tanimoto(x, y):

	d = np.dot(x, y.T)

	lx = np.vstack(tuple([np.sum(a) * np.ones(d.shape[1]) for a in x]))
	ly = np.vstack(tuple([np.sum(a) * np.ones(d.shape[0]) for a in y])).T


	return d / (lx + ly - d)