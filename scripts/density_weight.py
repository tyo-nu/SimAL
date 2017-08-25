__author__ = 'Dante'

import math
import pybel
import re
from structures import fingerprinter as fptr


def avg_proximity(groupa, groupb, f=fptr.integer_sim):

	all_tcs = []

	for x in groupa:
		tcs = [f(x, y) for y in groupb]
		all_tcs.append(tcs)

	return [sum(z) / len(all_tcs) for z in all_tcs]


def entropy(p_pos):

	return -math.fsum([p_pos * math.log(p_pos, 2), (1 - p_pos) * math.log(1 - p_pos, 2)])

def hyper_distance(d):

	if d != 0:
		return math.log10(1 / math.fabs(d))

	else:
		return 0

def weight(x, avg, beta=1):
	"""Calculates density weights for unlabelled points based on their relative centrality in the distribution of
	unlabelled points. Distance can be distance from the hyperplane or a shannon entropy.  Set shannon to True if it is
	the latter."""

	xid = x * (avg ** beta)

	return math.fabs(float(xid))


def generate_report(results, outfile='dw_scores.sdf', vendors_list=None, zinc=True):

	if vendors_list:
		sd = pybel.Outputfile('sdf', "ltd_" + outfile, overwrite=True)
	else:
		sd = pybel.Outputfile('sdf', outfile, overwrite=True)

	if zinc:
		for j, (smiles, xid, label, vendors, zincid) in enumerate(results):
			mol = pybel.readstring('smi', str(smiles))
			mol.data['ZINCID'] = zincid
			mol.data['Smiles'] = smiles
			mol.data['Rank'] = j + 1
			mol.data['x*'] = xid
			mol.data['Label'] = label
			mol.data['log P'] = mol.calcdesc(descnames=['logP'])['logP']
			if vendors_list:
				v = [re.compile(vend) for vend in vendors_list]
				approved = []
				for patt in v:
					for co in list(set([b for c in [x.keys() for x in vendors] for b in c])):
						if re.search(patt, co):
							approved.append(co)

				mol.data['vendors'] = '\n'.join(list(set(approved)))
				if len(mol.data['vendors']) > 0:
					sd.write(mol)

			else:
				mol.data['vendors'] = '\n'.join(list(set([b for c in [x.keys() for x in vendors] for b in c])))
				sd.write(mol)

		sd.close()

	else:
		for j, (smiles, xid, label) in enumerate(results):
			mol = pybel.readstring('smi', str(smiles))
			mol.data['x*'] = xid
			mol.data['Label'] = label

			sd.write(mol)
		sd.close()



