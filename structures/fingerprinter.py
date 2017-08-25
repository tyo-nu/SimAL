__author__ = 'Dante'

import json
import pybel
from rdkit import Chem
import os
import os.path
import copy
import numpy as np
from numpy import linalg as LA
import math
from random import choice
from scipy.cluster import hierarchy as hcluster
from chem import chem


CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")


def fp_bits(x):

	if '@' in x:
		molIn = pybel.readstring('smi', x)
		obmol = molIn.OBMol
		obmol.AddHydrogens()
		mol = pybel.Molecule(obmol)
	else:
		mol = pybel.readstring('smi', x)

	try:
		fp = mol.calcfp('FP4')
		bitsOn = fp.bits
	except IOError:
		bitsOn = [0]

	return bitsOn


def reconstruct_fp(s, fptype='FP4'):
	'''This function returns a bit-vector based on activated bits.'''
	fpL = {'FP4': 308, 'FP2': 1024, 'FP3': 55, 'MACCS': 166}

	mol = pybel.readstring('smi', str(s))
	fp = mol.calcfp(fptype)
	bitlist = fp.bits

	if 0 in bitlist:
		rawfp = None

	else:
		rawfp = [0] * (fpL[fptype] + 1)

		for b in bitlist:
			rawfp[b] += 1

	return rawfp


def integer_fp(smile, smartsqfile='SMARTSFileFull.json'):
	'''Creates an integer-valued fingerprint.  Probably takes longer since it is a VERY hacked version, but it generates
	data for SVMs.'''

	rdmol = Chem.MolFromSmiles(smile)

	if rdmol:
		smartsData = open(os.path.join(CHEMPATH, smartsqfile))
		encodedSmartsDict = json.load(smartsData)
		smartsData.close()
		smartsDict = {k.encode('utf-8'): v.encode('utf-8').split(': ')[1].strip(' \n') for k, v in encodedSmartsDict.iteritems()}

		fpt = [0] * len(smartsDict.keys())

		cpd = pybel.readstring('smi', smile)

		for k, v in smartsDict.iteritems():
			query = pybel.Smarts(v)
			fpt[int(k) - 1] = len(query.findall(cpd))

		#SMARTS querying directly for chirality does not give RS isomerism, so adding it at the end.  May get flexed to
		#_check_reverse for consistency.
		rect, sini = chem.flag_chiral(rdmol)

		fpt.append(rect)
		fpt.append(sini)

		return fpt

	else:

		return None


def bin_array(narray):

	bin = []

	for row in narray:
		bin.append(np.array([x if x == 0 else 1 for x in row]))

	return np.vstack(tuple(bin))


def pare_fp(fpt, bitAddresses):
	'''Pares down a full fpt to conform with a feature-reduced fpt whose bitAddresses dictionary is given.'''
	if len(fpt) < max(bitAddresses.values()):
		raise IOError('Fingerprint argument has too few bits to accommodate all of the addressed bits.')

	rawFpt = []
	for i in sorted(bitAddresses.values()):
		rawFpt.append(fpt[i - 1])

	return rawFpt


def integer_sim(a, b):
	if not isinstance(a, np.ndarray):
		a = np.array(a)
	if not isinstance(b, np.ndarray):
		b = np.array(b)

	if np.sum(a) == 0 and np.sum(b) == 0:
		t = 0
	else:
		t = float(np.dot(a, b)) / float((LA.norm(a, 2) ** 2 + LA.norm(b, 2) ** 2 - np.dot(a, b)))

	return t

def bin_tc(a, b):
	a = np.array([x if x == 0 else 1 for x in a])
	b = np.array([x if x == 0 else 1 for x in b])

	if np.sum(a) == 0 and np.sum(b) == 0:
		t = 0
	else:
		t = float(np.dot(a, b)) / float((np.sum(a) + np.sum(b)) - np.dot(a, b))

	return t


def assemble_sample_array(fpList):
	'''Assembles a n_samples by n_features array.'''

	fpTup = tuple(fpList)

	sampleArray = np.vstack(fpTup)

	return sampleArray


class FingerprintArray:
	'''Class for array of training data of biochemicals.'''

	def __init__(self, dataset):

		self.nArray = dataset
		self.__origArray = dataset
		self.bitAddresses = {}

	def reset_array(self):

		self.nArray = copy.deepcopy(self.__origArray)

	def describe_bits(self, descfile):

		f = open(descfile)
		encodedBitNames = json.load(f)
		f.close()
		bitNames = {k.encode('utf-8'): v.encode('utf-8') for k, v in encodedBitNames.iteritems()}

		descriptors = {k: bitNames[v] for k, v in self.bitAddresses.iteritems()}

		with open('bits.json', 'w') as g:
			g.write(json.dumps(descriptors, indent=4))

	def shannon_entropy_calc(self, tol=0, report=False):
		goodBitList = []
		goodBits = []

		for x in range(self.nArray.shape[1]):
			samples = float(self.nArray[:, x].size)
			onInstances = [b for b in self.nArray[:, x] if b > 0]
			pOn = []

			for num in set(onInstances):
				p_i = onInstances.count(num) / samples
				pOn.append(p_i)

			pOff = 1 - math.fsum(pOn)

			try:
				entropy = -(np.sum(np.multiply(np.array(pOn), np.log2(np.array(pOn)))) + pOff * math.log(pOff, 2))
			except ValueError:
				entropy = 0

			if entropy > tol:
				goodBitList.append(x)
				goodBits.append(self.nArray[:, x])

		self.nArray = np.vstack(tuple(goodBits)).T

		if len(self.bitAddresses) == 0:
			self.bitAddresses = {k: v for k, v in zip(range(self.nArray.shape[1]), goodBitList)}
		else:
			updateGoodBits = [self.bitAddresses[x] for x in goodBitList]
			self.bitAddresses = {k: v for k, v in zip(range(self.nArray.shape[1]), sorted(updateGoodBits))}

		if report:
			return self.nArray.shape, self.bitAddresses

	def calc_clusters(self, tol=0.02, report=False):

		#We use the transpose of the training matrix in order to cluster features on their pairwise correlation throughout the dataset.
		try:
			clusters = hcluster.fclusterdata(self.nArray.T, tol, criterion='distance', metric='correlation', method='average')
		except ValueError:
			Z = hcluster.linkage(self.nArray.T, method='average', metric='correlation')
			np.clip(Z, 0, 10000, out=Z)
			clusters = hcluster.fcluster(Z, tol, criterion='distance')

		clusterDict = {x: [] for x in list(set(clusters))}

		for i, x in enumerate(clusters):
			clusterDict[x].append(i)

		goodBitList = []

		for k, v in clusterDict.iteritems():
			if len(v) > 1:
				goodBitList.append(choice(v))
			elif len(v) == 1:
				goodBitList.append(v[0])
			else:
				continue

		goodBitList.sort()

		goodBits = [self.nArray[:, x] for x in goodBitList]

		self.nArray = np.vstack(tuple(goodBits)).T

		if len(self.bitAddresses) == 0:
			self.bitAddresses = {k: v for k, v in zip(range(self.nArray.shape[1]), goodBitList)}
		else:
			updateGoodBits = [self.bitAddresses[x] for x in goodBitList]
			self.bitAddresses = {k: v for k, v in zip(range(self.nArray.shape[1]), sorted(updateGoodBits))}

		if report:
			return self.nArray.shape, self.bitAddresses

	def generate_pybel_fpfile(self, flag=False):

		if flag:
			if os.path.isfile("SMARTSFileFull.json"):
				bitsData = open("SMARTSFileFull.json")
				encodedBitsDict = json.load(bitsData)
				bitsData.close()
				bitsDict = {k.encode('utf-8'): v.encode('utf-8') for k, v in encodedBitsDict.iteritems()}

				toFp = [bitsDict[str(x)] for x in sorted(self.bitAddresses.values())]

				f = open('FPFile.txt', 'w')
				for y in toFp:
					f.write(y)
				f.close()

			else:
				raise IOError('The file SMARTSFileFull.json must be in the current directory to use this function.')

		else:
			pass


# if __name__ == "__main__":
# 	usage = "Usage: %prog [options] trainingset"
# 	parser = OptionParser(usage=usage)
# 	parser.add_option('-c', '--clustertol', dest='clustertol', type='float', default=0.05, help='clustering criterion')
# 	parser.add_option('-f', '--fp4file', dest='fp4file', action='store_true', help='generate a pybel fp4 file')
# 	parser.add_option('-r', '--reportstats', dest='reportstats', action='store_true', help='show stats for reduced arrays')
# 	parser.add_option('-s', '--shannontol', dest='shannontol', type='float', default=0, help='entropy rejection criterion')
# 	(options, args) = parser.parse_args()
#
# 	smilesData = open(args[0])
# 	encodedSmilesDict = json.load(smilesData)
# 	smilesData.close()
# 	smilesDict = {k.encode('utf-8'): v.encode('utf-8') for k, v in encodedSmilesDict.iteritems()}
#
# 	smiles = smilesDict.values()
#
# 	fpList = []
#
# 	for z in smiles:
# 		try:
# 			f = integer_fp(z)
# 		except IOError:
# 			f = None
# 		fpList.append(f)
#
# 	kDataset = FingerprintArray(assemble_sample_array(fpList))
#
# 	# elapsed = time.time() - start
# 	# print "Time elapsed: %s seconds." % elapsed
#
# 	entro_s, entro_b = kDataset.shannon_entropy_calc(tol=options.shannontol, report=True)
#
# 	print entro_s, entro_b


