__author__ = 'Dante'

import os
import random
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
from rdkit import Chem
from pymongo import MongoClient
import fingerprinter as fptr
from utils import stattests

CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")
COLL = MongoClient().zincDB.zinc_pages
EXC = [1, 2, 3, 4, 307, 295, 296, 297, 298, 299]

class Priority(object):

	def __init__(self, filename, org=None, ec=None):

		__full_path = os.path.join(CHEMPATH, filename)

		self.__supplier = Chem.SDMolSupplier(__full_path)
		self.smiles = [mol.GetProp("Smiles") for mol in self.__supplier if mol is not None]
		self.zinc_ids = [mol.GetProp("ZINCID") for mol in self.__supplier if mol is not None]

		#self.fps = [COLL.find({"_id": z}, {"fp": ""})["fp"] for z in self.zinc_ids]
		self.fps = [fptr.integer_fp(smi) for smi in self.smiles]
		self.matrix_ = np.vstack(tuple([np.array(fpt) for fpt in self.fps]))

		self.occurrence = {i: np.count_nonzero(col) / col.size for i, col in enumerate(self.matrix_.T)}

	def off_bits(self):

		off = []

		for i, col in enumerate(self.matrix_.T):
			if np.sum(col) == 0:
				off.append(i)

		return off

	def test_enrichment(self, s_level=0.05, etest=stattests.barnard_exact):

		split_index = int(0.1 * self.matrix_.shape[0])
		top_ = self.matrix_[:split_index]
		pool_ = range(self.matrix_.shape[0])
		rand_indexes = random.sample(pool_, split_index + 1)
		rand_ = self.matrix_[rand_indexes]
		#bottom_ = self.matrix_[split_index:]

		test_values = []

		for j in range(self.matrix_.shape[1]):
			top_on_occurrences = np.count_nonzero(top_.T[j])
			top_off_occurrences = top_.shape[0] - top_on_occurrences

			rand_on_occurrences = np.count_nonzero(rand_.T[j])
			rand_off_occurrences = rand_.shape[0] - rand_on_occurrences

			if top_on_occurrences + rand_on_occurrences == 0 or top_off_occurrences == 0 or rand_off_occurrences == 0:
				continue

			contin_table = np.array([[top_on_occurrences, top_off_occurrences], [rand_on_occurrences, rand_off_occurrences]])

			#Effect size CI calculations. The negative sign is used because the order of the table is inverted, so we make
			#instances where top_on > rand_on positive rather than negative
			risk_diff = -((float(rand_on_occurrences) / float(rand_on_occurrences + rand_off_occurrences)) - (float(top_on_occurrences) / float(top_off_occurrences + top_on_occurrences)))
			p_hat = float(np.sum(contin_table[0])) / float(np.sum(contin_table))
			rdse = (p_hat * (1 - p_hat) * ((1 / float(np.sum(contin_table[:, 0]))) + (1 / float(np.sum(contin_table[:, 1]))))) ** 0.5
			ci_lower = risk_diff - 1.96 * rdse
			ci_upper = risk_diff + 1.96 * rdse

			p_val = etest(contin_table)
			expected_table = chi2_contingency(contin_table)[3]
			# Since this is a two-sided test, a significant p-value will crop up for feature that is either enriched or
			# unenriched in the top x percentile.  We find an expected value and subtract the number of on occurrences
			# from it.  If this difference is positive, we can tell that the top compounds are unenriched, and if it is
			# negative, we can tell that the top compounds are enriched for that feature at a significant rate.
			exp_diff = round(expected_table[0, 0], 2) - top_on_occurrences

			test_values.append((j + 1, p_val, risk_diff, ci_lower, ci_upper, exp_diff))

		f = open('enrich_rank.txt', 'w')
		for v in sorted(test_values, key=lambda y: y[2]):
			f.write(str(v) + '\n')
		f.close()

		print "ENRICHED, p < alpha"
		print sorted([x for x in test_values if x[1] < s_level / self.matrix_.shape[1] and -x[5] > 0 and x[0] not in EXC], key=lambda y: y[1])
		print "UNENRICHED, p < alpha"
		print sorted([x for x in test_values if x[1] < s_level / self.matrix_.shape[1] and -x[5] < 0 and x[0] not in EXC], key=lambda y: y[1])
		print "OTHER ENRICHED"
		print sorted([x for x in test_values if x[1] < s_level and -x[5] > 0 and x[0] not in EXC], key=lambda y: y[1])
		print "OTHER UNENRICHED"
		print sorted([x for x in test_values if x[1] < s_level and -x[5] < 0 and x[0] not in EXC], key=lambda y: y[1])

if __name__ == "__main__":

	p = Priority('MenD_zincAL.sdf')
	p.test_enrichment()
