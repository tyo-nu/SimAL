__author__ = 'Dante'

from reaction import BrendaReaction
from reaction import SDFReaction
from databases import db_queries as dbq
import numpy as np
from rdkit import Chem
from rdkit.Chem import MCS
import fingerprinter as fptr
import pybel
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as acc
import hashlib
import re
import json
import os
import random
import time
from sklearn import cross_validation as cv
from scripts import density_weight as dw
from collections import Counter
from utils import plots
from scipy.spatial.distance import pdist, squareform

"""Classes and functions for dealing with large scale Reactions."""

CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")
LOC = os.getcwd()


def fp_str_to_array(rxn_fps):

	all_fps = []

	for fpt in [(fp.strip('[] ').replace('\n', ' ').split()) for fp in rxn_fps]:
		all_fps.append(np.array([int(x) for x in fpt]))

	return all_fps


def count_bits(rxn_fps):

	all_fps = fp_str_to_array(rxn_fps)

	narray = np.vstack(tuple(all_fps))

	redundants = [1, 2, 3, 4, 12, 88, 287, 288, 295, 300, 301, 302, 307]  #Bits that are double covered

	if not isinstance(narray, np.ndarray):
		raise IOError("fps not a numpy array.")

	samples = float(narray.shape[0])

	_bits = []

	for x in range(narray.shape[1]):
		consumed = [b for b in narray[:, x] if b < 0]
		p_consumed = len(consumed) / samples

		if p_consumed > 0:

			_bits.append((x, p_consumed))

	_bits.sort(key=lambda z: z[1], reverse=True)

	return [b + 1 for b, x in _bits if b + 1 not in redundants]


def _check_reverse_(rxn_list, racemic=False):
	"""Assembles separate listings for reactions that are inverses of one another within the same Isozyme listing."""

	sini = []
	rect = []

	if racemic:
		#This method of sorting may result in fewer compounds; BRENDA occasionally has reactions incorrectly annotated
		#that products go to products, producing a zero vector for the fingerprint, which is what is used for sorting.
		for i, rxn in enumerate(rxn_list):
			if rxn.fpt_all_()[0][-1] < 0:
				sini.append(i)
				print rxn.fpt_all_()[0][-2], rxn.fpt_all_()[0][-1]
			elif rxn.fpt_all_()[0][-2] < 0:
				rect.append(i)
			else:
				pass

		return [rect, sini]

	else:

		#Reaction species combination to test for uniqueness of substrates.

		all_reactions = {frozenset(r.reactants + r.products): [] for r in rxn_list}

		for i, rxn in enumerate(rxn_list):
			all_reactions[frozenset(rxn.reactants + rxn.products)].append(i)

		#Now check to see if reactions with duplicate reaction + product combinations are truly reverses of each other,
		#or duplicates in the same direction.  We do this by hashing the reactants + products list in that order for each
		#duplicate in all_reactions.  If the length of the resulting hash list is greater than 1 (it ideally should be
		#either 1 or 2), then there exists a reversed pair of reactions and we can then proceed to the clustering algorithm.

		reverse_count = 0

		for rxns in all_reactions.values():
			reaction_hashes = []
			if len(rxns) > 1:
				for r in rxns:
					h = hashlib.sha1(str(rxn_list[r].reactants + rxn_list[r].products)).hexdigest()
					reaction_hashes.append(h)
			if len(set(reaction_hashes)) > 1:
				reverse_count += 1

		#If there are reversed reactions, we then need to sort them.  This ensures that in a reaction that proceeds, say,
		#alcohol <==> aldehyde, we sort the "forward" reactions from the "reverse" such that all reactant groups in one
		#group contain the aldehyde, and all reactant groups in the other group contain the alcohol.  This does NOT assign
		#a canonical direction to the reaction in question; it only divides them into the appropriate groups.  What those
		#groups are called is left ot the user to determine.  This assumes that a listing contains reactions going in
		#opposing directions iff two opposite reactions occur in the listing.  In theory, this is not strictly true,
		#but since this is common, we can make this assertion in order to avoid clustering on every Isozyme.

		if reverse_count > 0:

			#Assemble X matrix for agglomerative clustering.

			all_fps = np.vstack(tuple([np.array(r.fpt_all_()[0]) for r in rxn_list]))

			#Agglomerative clustering algorithm.  We insist on 2 clusters ("forward" & "reverse", but again, this code
			#does not identify which cluster is canonically "forward" or "reverse"), with a Euclidean linkage with minimized
			#variance.

			clust = AC(n_clusters=2)
			labels = clust.fit_predict(all_fps)

			return [[x for x, l in enumerate(labels) if l == 0], [x for x, l in enumerate(labels) if l == 1]]

		else:

			return None


class Isozyme(object):
	"""Top level isozyme class"""

	def __init__(self):

		self.org = None
		self.ec = None

		self.reactions = []
		self.orphan_reactions = []
		self.reaction_fingerprints = []
		self.r = []
		self.p = []

		self.rxt_groups = {}
		self.rxf_groups = {}
		self.rxt_groups_f = {}

		self.q = {}
		self.pos = {}
		self.neg = {}

	def has_complete_reactions(self):
		"""Helper function that checks for complete reactions.  If the reactions are of the type that adds a nominal
		metabolite to a macromolecule (i.e., acetylates DNA), then the product will not have any smiles since
		macromolecules cannot be converted under the current scheme."""
		if len(self.reactions) > 0:
			return True
		else:
			return False

	def analyze_reactions(self, smarts_file='SMARTSFileFull.json'):

		redundants = [1, 2, 3, 4, 12, 88, 287, 288, 295, 300, 301, 302, 307]  # Bits that are double covered
		f = open(os.path.join(CHEMPATH, smarts_file))
		patterns = {k.encode('utf-8'): v.encode('utf-8') for k, v in json.load(f).iteritems()}
		f.close()

		bit_q = {}

		for k, v in self.rxt_groups.iteritems():
			# Uses rdkit to look for a maximum common subgraph (MCS).
			mols = [Chem.MolFromSmiles(str(mol)) for mol in list(v)]
			try:
				mcs = MCS.FindMCS(mols, matchValences=True, bondCompare='bondtypes', ringMatchesRingOnly=True, completeRingsOnly=True)
				query = mcs.smarts
				atom_count = mcs.numAtoms
			except:
				atom_count = 0

			bits_from_rxns = count_bits(self.rxf_groups[k])

			if atom_count > 1:
				self.q[k] = [query]

				#If there is an MCS in the group of molecules, we can reconstruct a representative molecule from the SMARTS
				#query and fingerprint it.  Cross-referencing with the bits present in the reaction FP give search criteria
				#that can be used to find negative or test set molecules that don't necessarily have the MCS, but are
				#needed to make an equivalently sized negative data set

				pattern = Chem.MolFromSmarts(query)
				pattern_fp = fptr.integer_fp(Chem.MolToSmiles(pattern))

				in_pattern = [patterns[str(i + 1)].split(': ')[1].strip('\n') for i, x in enumerate(pattern_fp) if x > 0 and i + 1 not in redundants and i + 1 in bits_from_rxns]

				if len(in_pattern) > 0:

					bit_q[k] = [(b, patterns[str(b)].split(': ')[1].strip('\n')) for b in bits_from_rxns if b < 311]

				else:
					#This is a list of tuples for base SMARTS queries detected. Numbers are for ZINC queries, actual
					#patterns are for negative set construction, if needed
					bit_q[k] = [(b, patterns[str(b)].split(': ')[1].strip('\n')) for b in bits_from_rxns if b < 311]

			else:

				bit_q[k] = [(b, patterns[str(b)].split(': ')[1].strip('\n')) for b in bits_from_rxns if b < 311]

		return bit_q

	def show_sets(self, i):

		outname = 'report' + str(i) + '.sdf'
		output = pybel.Outputfile('sdf', outname, overwrite=True)

		for s in self.rxt_groups[i]:

			mol = pybel.readstring('smi', str(s))
			output.write(mol)

		output.close()

	def tc_matrix(self, k, f=fptr.integer_sim):

		x_pos_array = np.vstack(tuple([t[1] for t in self.pos[k]]))
		x_neg_array = np.vstack(tuple([t[1] for t in self.neg[k]]))
		x_array = np.vstack((x_pos_array, x_neg_array))
		y_obj = []
		y_obj += [1] * x_pos_array.shape[0]
		y_obj += [-1] * x_neg_array.shape[0]
		smiles_access = [t[0] for t in self.pos[k]] + [t[0] for t in self.neg[k]]
		n_smiles = max([len(x) for x in smiles_access])
		y_array = np.array(zip(y_obj, smiles_access), dtype=[('label', 'i4'), ('smiles', '|S%s' % str(n_smiles))])

		cmatrix = pdist(x_array, 'jaccard')
		sqmatrix = np.ones((len(y_obj), len(y_obj))) - squareform(cmatrix)

		outfile_heat = "sim_%s_%s" % (self.org, self.ec)

		plots.similarity_matrix(smiles_access, sqmatrix, y_array, outfile_heat, yl='Compounds', zl='Similarity Score', num_labels=True)

	def off_bits(self, k):

		x_pos_array = np.vstack(tuple([t[1] for t in self.pos[k]]))
		x_neg_array = np.vstack(tuple([t[1] for t in self.neg[k]]))
		x_array = np.vstack((x_pos_array, x_neg_array))

		off = []

		for i, col in enumerate(x_array.T):
			if np.sum(col) == 0:
				off.append(i)

		return off

	def random_negatives(self, k):

		from_seed = dbq.rand_kegg(len(self.pos[k]))
		add_in = [(x, np.array(fptr.integer_fp(x))) for x in from_seed]

		self.neg[k] = add_in

	def add_from_sdf(self, f, k, pos=True):

		add_in = []
		pos_smiles = [x for x, y in self.pos[k]]

		sdf_in = pybel.readfile('sdf', os.path.join(CHEMPATH, f))
		for mol in sdf_in:
			smi = mol.write('can').strip()
			fp = fptr.integer_fp(smi)
			add_in.append((smi, np.array(fp)))

		if k in self.pos and pos:
			for entry in add_in:
				if entry[0] not in pos_smiles:
					self.pos[k].append(entry)
		elif k in self.neg and not pos:
			for entry in add_in:
				if entry[0] not in pos_smiles:
					self.neg[k].append(entry)
		elif k not in self.neg and not pos:
			self.neg[k] = []
			for entry in add_in:
				if entry[0] not in pos_smiles:
					self.neg[k].append(entry)
		else:
			self.pos[k] = add_in

		self.mass_avg = [np.mean([pybel.readstring('smi', str(s)).exactmass for s in list(set(l))]) for k, l in self.rxt_groups.iteritems()]
		self.mass_std = [np.std([pybel.readstring('smi', str(s)).exactmass for s in list(set(l))]) for k, l in self.rxt_groups.iteritems()]

	def add_from_smiles(self, smiles, k, cls):

		if cls not in {1, -1}:
			raise IOError('Valid classes are 1 and -1')

		add_in = (pybel.readstring('smi', smiles).write('can').strip(), np.array(fptr.integer_fp(smiles)))

		if k in self.pos and cls == 1:
			if add_in not in self.pos[k]:
				self.pos[k].append(add_in)
		elif k in self.neg and cls == -1:
			if add_in not in self.neg[k]:
				self.neg[k].append(add_in)
				if add_in in self.pos[k]:
					self.pos[k].remove(add_in)
		elif k not in self.neg and cls == -1:
			self.neg[k] = add_in
			if add_in in self.pos[k]:
				self.pos[k].remove(add_in)
		else:
			pass

		self.mass_avg = [np.mean([pybel.readstring('smi', str(s)).exactmass for s in list(set(l))]) for k, l in self.rxt_groups.iteritems()]
		self.mass_std = [np.std([pybel.readstring('smi', str(s)).exactmass for s in list(set(l))]) for k, l in self.rxt_groups.iteritems()]

	def xval_selection(self, k, beta=1.0, batch=1, kernel='rbf', random_seed=None, iterations=1000, initial=2, c=1.0, gamma=0.005, degree=3, decf=True, simfp=fptr.integer_sim):

		outfile_xval = "al_xval_%s_%s_beta%s_batch%s_%s_rseed%s" % (self.org, self.ec, str(beta).replace('.', ''), str(batch), kernel, str(random_seed))
		outfile_heat = "al_ins_%s_%s_beta%s_batch%s_%s_rseed%s" % (self.org, self.ec, str(beta).replace('.', ''), str(batch), kernel, str(random_seed))

		rand_res = []
		dw_res = []

		x_pos_array = np.vstack(tuple([t[1] for t in self.pos[k]]))
		x_neg_array = np.vstack(tuple([t[1] for t in self.neg[k]]))
		x_array = np.vstack((x_pos_array, x_neg_array))
		y_obj = []
		y_obj += [1] * x_pos_array.shape[0]
		y_obj += [-1] * x_neg_array.shape[0]
		smiles_access = [t[0] for t in self.pos[k]] + [t[0] for t in self.neg[k]]
		n_smiles = max([len(x) for x in smiles_access])
		rankings = {k_: [] for k_ in smiles_access}
		y_array = np.array(zip(y_obj, smiles_access), dtype=[('label', 'i4'), ('smiles', '|S%s' % str(n_smiles))])

		for iteration in range(iterations):

			# If you set the random_state kwarg, it will make the test set uniform across all iterations.
			x_train, x_test, y_train, y_test = cv.train_test_split(x_array, y_array, test_size=0.4, random_state=random_seed)

			i = range(x_train.shape[0])
			#Change the second argument to start with a different number of samples. The while loop ensures that the initial
			#training sample is instantiated with at least one example from each class.
			rand_train = random.sample(i, initial)

			while set(y_train[rand_train]['label']) != {-1, 1}:
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

				clf_rand = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree, probability=not decf)
				clf_dw = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree, probability=not decf)

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
				available_dw_ = list(set(i) - set(dw_train))
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

			clf_rand_last = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree).fit(x_train[rand_train], y_train['label'][rand_train])
			clf_dw_last = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree).fit(x_train[dw_train], y_train['label'][dw_train])
			n.append(len(rand_train))
			r_last = clf_rand_last.predict(x_test)
			d_last = clf_dw_last.predict(x_test)
			rand_scores.append(acc(y_test['label'], r_last))
			dw_scores.append(acc(y_test['label'], d_last))
			rand_res.append(np.array(rand_scores))
			dw_res.append(np.array(dw_scores))

		final = {k: Counter(v) for k, v in rankings.iteritems()}
		positions = [j + 1 for j in range(max([x for val in rankings.values() for x in val]))]
		data_ = []
		for cpd, ctr in final.iteritems():
			data_.append(np.array([ctr[pos] for pos in positions]))

		data_in_ = np.vstack(tuple(data_)).T

		plots.dw_rand_curves(n, rand_res, dw_res, outfile_xval, iterations=iterations, c=c, decf=decf)
		# plots.dw_heatmap(positions, smiles_access, data_in_, y_array, outfile_heat, c=c, decf=decf)

		return dw_res

	def xval_selection_random(self, k, beta=1.0, batch=1, kernel='rbf', random_seed=None, iterations=1000, initial=2, c=1.0, gamma=0.005, degree=3, decf=True, simfp=fptr.integer_sim):

		outfile_xval = "al_xvalrandom_%s_%s_beta%s_batch%s_%s_rseed%s" % (self.org, self.ec, str(beta).replace('.', ''), str(batch), kernel, str(random_seed))

		rand_res = []
		dw_res = []

		for iteration in range(iterations):
			while True:
				try:
					self.random_negatives(k)

					x_pos_array = np.vstack(tuple([t[1] for t in self.pos[k]]))
					x_neg_array = np.vstack(tuple([t[1] for t in self.neg[k]]))

					x_array = np.vstack((x_pos_array, x_neg_array))
				except ValueError:
					continue
				break

			y_obj = []
			y_obj += [1] * x_pos_array.shape[0]
			y_obj += [-1] * x_neg_array.shape[0]
			y_array = np.array(y_obj)

			# If you set the random_state kwarg, it will make the test set uniform across all iterations.
			x_train, x_test, y_train, y_test = cv.train_test_split(x_array, y_array, test_size=0.4, random_state=random_seed)

			i = range(x_train.shape[0])
			# Change the second argument to start with a different number of samples. The while loop ensures that
			# the initial
			#training sample is instantiated with at least one example from each class.
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

			#Each data point in each iteration is created here.
			while len(rand_train) < x_train.shape[0]:

				clf_rand = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree, probability=not decf)
				clf_dw = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree, probability=not decf)

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
					adds = [ele[0] for ele in foo[:batch]]
					dw_train += adds
				elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
					dw_train += available_dw_
				else:
					pass

			clf_rand_last = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree).fit(x_train[rand_train], y_train[rand_train])
			clf_dw_last = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree).fit(x_train[dw_train], y_train[dw_train])
			n.append(len(rand_train))
			r_last = clf_rand_last.predict(x_test)
			d_last = clf_dw_last.predict(x_test)
			rand_scores.append(acc(y_test, r_last))
			dw_scores.append(acc(y_test, d_last))
			rand_res.append(np.array(rand_scores))
			dw_res.append(np.array(dw_scores))

		plots.dw_rand_curves(n, rand_res, dw_res, outfile_xval, iterations=iterations, c=c, decf=decf)

	def expval_selection(self, k, excl, initial=2, iterations=100, beta=1.0, c=1.0, gamma=0.005, degree=2, kernel='rbf', batch=1, decf=False, seed=None, simfp=fptr.integer_sim):

		rand_res = []
		dw_res = []

		k_dict = {'rbf': 'rbf', 'poly': 'poly'}

		outfile_expval = "al_expval%s_%s_beta%s_batch%s_%s_rseed%s" % (self.org, self.ec, str(beta).replace('.', ''), str(batch), kernel, str(seed))

		smiles_access = [t[0] for t in self.pos[k]] + [t[0] for t in self.neg[k]]
		n = max([len(str(x)) for x in smiles_access])

		x_pos_array = np.vstack(tuple([t[1] for t in self.pos[k]]))
		x_neg_array = np.vstack(tuple([t[1] for t in self.neg[k]]))

		y_obj = []

		y_obj += [1] * x_pos_array.shape[0]
		y_obj += [-1] * x_neg_array.shape[0]

		x_array = np.vstack((x_pos_array, x_neg_array))
		y_array = np.array(zip(y_obj, smiles_access), dtype=[('label', 'i4'), ('smiles', '|S%s' % str(n))])

		# Iterates n times.
		for iteration in range(iterations):
			# If you set the random_state kwarg, it will make the test set uniform across al iterations.
			x_train, x_test, y_train, y_test = cv.train_test_split(x_array, y_array, test_size=0.4, random_state=seed)

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

				clf_rand = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)
				clf_dw = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)

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
						xi = [dw.weight(dw.hyper_distance(clf_dw.decision_function(x_train[a])),
						                pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
					else:
						xi = [dw.weight(dw.entropy(clf_dw.predict_proba(x_train[a])[:, 0]),
						                pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
					foo = sorted(zip(available_dw_, xi), key=lambda x: x[1], reverse=True)
					dw_train += [ele[0] for ele in foo[:batch]]
				elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
					dw_train += available_dw_
				else:
					pass

			clf_rand_last = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[rand_train],
			                                                                                y_train['label'][
				                                                                                rand_train])
			clf_dw_last = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[dw_train],
			                                                                              y_train['label'][dw_train])
			n.append(len(rand_train))
			r_last = clf_rand_last.predict(x_test)
			d_last = clf_dw_last.predict(x_test)
			rand_scores.append(acc(y_test['label'], r_last))
			dw_scores.append(acc(y_test['label'], d_last))

			rand_res.append(np.array(rand_scores))
			dw_res.append(np.array(dw_scores))

		plots.dw_rand_curves(n, rand_res, dw_res, outfile_expval, iterations=iterations, c=c, decf=decf)

	def expval_selection_random(self, k, exp, initial=2, iterations=100, beta=1.0, c=1.0, gamma=0.005, degree=2, kernel='rbf', batch=1, decf=False, seed=None, simfp=fptr.integer_sim):

		rand_res = []
		dw_res = []

		k_dict = {'rbf': 'rbf', 'poly': 'poly'}

		outfile_expval = "al_expvalrandom_%s_%s_beta%s_batch%s_%s_rseed%s" % (self.org, self.ec, str(beta).replace('.', ''), str(batch), kernel, str(seed))

		suppl = pybel.readfile('sdf', os.path.join(CHEMPATH, exp))
		excl_pos = []
		excl_neg = []

		for mol in suppl:
			smi = mol.write('can').strip()
			cls = int(mol.data['label'])
			if cls == 1:
				self.add_from_smiles(smi, k, cls)
				excl_pos.append(smi)
			else:
				excl_neg.append(smi)

		excl = excl_neg + excl_pos

		# Iterates n times.
		for iteration in range(iterations):
			while True:
				try:
					self.random_negatives(k)
					for neg_smi in excl_neg:
						self.add_from_smiles(neg_smi, k, -1)

					smiles_access = [t[0] for t in self.pos[k]] + [t[0] for t in self.neg[k]]
					n_smiles = max([len(str(x)) for x in smiles_access])

					x_pos_array = np.vstack(tuple([t[1] for t in self.pos[k]]))
					x_neg_array = np.vstack(tuple([t[1] for t in self.neg[k]]))

					x_array = np.vstack((x_pos_array, x_neg_array))
				except ValueError:
					continue
				break

			y_obj = []
			y_obj += [1] * x_pos_array.shape[0]
			y_obj += [-1] * x_neg_array.shape[0]
			y_array = np.array(zip(y_obj, smiles_access), dtype=[('label', 'i4'), ('smiles', '|S%s' % str(n_smiles))])

			# If you set the random_state kwarg, it will make the test set uniform across al iterations.
			x_train, x_test, y_train, y_test = cv.train_test_split(x_array, y_array, test_size=0.4, random_state=seed)

			# Exclude experimental compounds from the random curve, and random compounds for the dw curve.
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

				clf_rand = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)
				clf_dw = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree, probability=True)

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
						xi = [dw.weight(dw.hyper_distance(clf_dw.decision_function(x_train[a])),
						                pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
					else:
						xi = [dw.weight(dw.entropy(clf_dw.predict_proba(x_train[a])[:, 0]),
						                pairwise_tc_avg[index_table_[a]], beta=beta) for a in available_dw_]
					foo = sorted(zip(available_dw_, xi), key=lambda x: x[1], reverse=True)
					dw_train += [ele[0] for ele in foo[:batch]]
				elif len(available_dw_) != 0 and len(available_dw_) % batch == len(available_dw_):
					dw_train += available_dw_
				else:
					pass

			clf_rand_last = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[rand_train],
			                                                                                y_train['label'][
				                                                                                rand_train])
			clf_dw_last = SVC(C=c, kernel=k_dict[kernel], gamma=gamma, degree=degree).fit(x_train[dw_train],
			                                                                              y_train['label'][dw_train])
			n.append(len(rand_train))
			r_last = clf_rand_last.predict(x_test)
			d_last = clf_dw_last.predict(x_test)
			rand_scores.append(acc(y_test['label'], r_last))
			dw_scores.append(acc(y_test['label'], d_last))

			rand_res.append(np.array(rand_scores))
			dw_res.append(np.array(dw_scores))

		plots.dw_rand_curves(n, rand_res, dw_res, outfile_expval, iterations=iterations, c=c, decf=decf)


class BrendaIsozyme(Isozyme):
	"""Specific class for constructing from Brenda data."""

	def __init__(self, org, ec, racemic=False):

		super(BrendaIsozyme, self).__init__()

		#Pulls data from BrendaDB.

		self.ec = ec
		self.org = org
		self.__res = dbq.pull_docs(org, ec)

		#Calls Reaction subclass to store each reaction in the DB listing.  Creates a list of reactions.

		for x in self.__res:
			try:
				rxn = BrendaReaction(x)
				if rxn.is_consistent():
					try:
						rxn.strip_cofactors()
					except ValueError:
						continue

					__fps = rxn.fpt_all_()
					self.reaction_fingerprints.append(__fps)

					self.reactions.append(rxn)
					self.r.append(rxn.r_fpt)
				else:
					self.orphan_reactions.append(rxn)
			except IOError:
				continue

		#Sort reactions that go in opposite directions (if opposites exist).
		self.directions = []

		reverse_flag = _check_reverse_(self.reactions, racemic=racemic)

		if reverse_flag:
			for i_list in reverse_flag:
				self.directions.append([self.reactions[i] for i in i_list])
		else:
			self.directions.append([x for x in self.reactions])

		self.__rxt_groups = {i: {j: list(rx) for j, rx in enumerate(zip(*[[s for s in r.reactants] for r in reaction]))} for i, reaction in enumerate(self.directions)}
		self.rxt_groups = {ki + kj: set([arr for arr in vj]) for ki, vi in self.__rxt_groups.iteritems() for kj, vj in vi.iteritems()}

		self.__rxf_groups = {i: {j: list(rx) for j, rx in enumerate(zip(*[[s for s in r.fpt_all_()] for r in reaction]))} for i, reaction in enumerate(self.directions)}
		self.rxf_groups = {ki + kj: [np.array_str(arr) for arr in vj] for ki, vi in self.__rxf_groups.iteritems() for kj, vj in vi.iteritems()}

		self.mass_avg = [np.mean([pybel.readstring('smi', str(s)).exactmass for s in list(set(l))]) for k, l in self.rxt_groups.iteritems()]
		self.mass_std = [np.std([pybel.readstring('smi', str(s)).exactmass for s in list(set(l))]) for k, l in self.rxt_groups.iteritems()]

		self.pos = {k: [(ss, fptr.integer_fp(str(ss))) for ss in list(v)] for k, v in self.rxt_groups.iteritems()}

	def negative_set(self, mmodel, additional_queries=False):

		n_smiles = {k: [] for k in self.rxt_groups.keys()}

		for k, v in self.rxt_groups.iteritems():
			if not additional_queries and k in self.q:
				suppl = Chem.SDMolSupplier(os.path.join(CHEMPATH, mmodel))
				patt = Chem.MolFromSmarts(self.q[k][0])
				matches = [Chem.MolToSmiles(x) for x in suppl if x.HasSubstructMatch(patt) and pybel.readstring('smi', Chem.MolToSmiles(x)).write('can').strip('\n\t') not in v]
				n_smiles[k] = matches
			elif additional_queries:
				n_smiles[k] = []
			else:
				n_smiles[k] = []

		return n_smiles

class SDFIsozyme(Isozyme):
	"""Specific class for constructing from experimental data in SDF format."""

	def __init__(self, sdffile, assay):

		super(SDFIsozyme, self).__init__()

		#Pulls data from sdf file.

		self.__res = pybel.readfile('sdf', sdffile)

		#Calls Reaction subclass to store each reaction in the DB listing.  Creates a list of reactions.

		for x in self.__res:
			rxn = SDFReaction(x, assay)
			if rxn.is_consistent():

				__fps = rxn.fpt_all_()

				self.reactions.append(rxn)
				self.reaction_fingerprints.append(__fps)

				self.r.append(rxn.r_fpt)
				self.p.append(rxn.p_fpt)


def MCS_analysis(fileout, frag):

	r = open(os.path.join(LOC, fileout), 'w')
	pattern = '^' + frag

	for entry in dbq.ec_in("Escherichia coli K12"):
		ec = entry
		if re.search(pattern, ec):
			try:
				a = BrendaIsozyme("Escherichia coli K12", ec)
				print ec, a.mass_avg
			except IOError:
				continue

			a.analyze_reactions()
			for v in [float(k) / len(a.rxt_groups[i]) for i, k in a.negative_set("iAF1260.sdf").iteritems()]:
				r.write(a.ec + ' ' + str(v) + '\n')

		else:
			pass

	r.close()

if __name__ == "__main__":

	a = BrendaIsozyme("Escherichia coli K12", "2.2.1.9")
	a.add_from_sdf("MenD_rd1.sdf", 0, pos=True)
	a.add_from_sdf("MenDNegatives.sdf", 0, pos=False)
	a.xval_selection(0, c=5)



