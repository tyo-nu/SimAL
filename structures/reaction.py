__author__ = 'Dante'

import json
import fingerprinter as fptr
from databases import db_queries as dbq
import numpy as np
import os.path

CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")

class Reaction(object):
	"""Class containing reaction information sourced from a BrendaDB query that outputs a tuple, rxn_db_page, with the
	form (reaction doc, list of cofactors docs).  Contains helper functions for mapping these reactions and calculating
	their fingerprints."""

	def __init__(self):

		self.reactant_names = []
		self.reactant_smiles = []
		self.product_names = []
		self.product_smiles = []

		self.stoichiometry = []

		self.reactants = []
		self.products = []
		self.stoich = []
		self.r_fpt = []
		self.p_fpt = []

	def is_consistent(self):
		"""Helper function to determine if all the names in the reaction have smiles."""

		reactants = len([x for x in self.reactant_names if x != ''])
		r_smiles = len([x for x in self.reactant_smiles if x != ''])

		products = len([x for x in self.product_names if x != ''])
		p_smiles = len([x for x in self.product_smiles if x != ''])

		if reactants == r_smiles and products == p_smiles:
			return True
		else:
			return False

	def is_many(self):
		"""Helper function that can flag a Reaction as many-to-one, one-to-many, or many-to-many."""

		if len(self.reactants) == 1 and len(self.products) == 1:
			return False
		else:
			return True

	def __is_valid(self):
		"""Ensures that there are compounds on both sides of the reaction."""

		rl = len(self.reactants)
		pl = len(self.products)
		sl = len(self.stoich)

		if rl > 0 and pl > 0 and sl == rl + pl:
			return True
		else:
			return False

	def fpt_all_(self):
		"""Calculates the fingerprints of each remaining reactant in the cofactor-free reaction."""
		#Are 2 fingerprints really needed here? Probably not, may to to revamp tests if correction made.
		if len(self.r_fpt) == 1 and len(self.p_fpt) == 1:
			return [self.p_fpt[0] - self.r_fpt[0]]

		elif len(self.r_fpt) == 1 and len(self.p_fpt) == 2:
			fps = []
			for p in self.p_fpt:
				fps.append(p - self.r_fpt[0])
			return fps

		elif len(self.r_fpt) == 2 and len(self.p_fpt) == 1:
			fps = []
			for r in self.r_fpt:
				fps.append(self.p_fpt[0] - r)
			return fps

		elif len(self.r_fpt) == 2 and len(self.p_fpt) == 2:
			return [p - r for p, r in zip(self.p_fpt, self.r_fpt)]

		else:
			raise IOError("Reaction greater than 2 -> 2")


class BrendaReaction(Reaction):

	def __init__(self, rxn_db_page):

		super(BrendaReaction, self).__init__()

		rxn_entry = rxn_db_page[0]
		cof_entry = rxn_db_page[1]

		self.reactant_names = rxn_entry["r_name"]
		self.reactant_smiles = rxn_entry["r_smiles"]
		self.product_names = rxn_entry["p_name"]
		self.product_smiles = rxn_entry["p_smiles"]

		self.stoichiometry = rxn_entry["s"]

		# Maybe redundant code, consider deprecation if is already covered by cof/pair/small files.
		if len(cof_entry) > 0:
			self.cofactor_names = [x["name"] for x in cof_entry]
			self.cofactor_smiles = [x["smiles"] for x in cof_entry]
		else:
			self.cofactor_names = []
			self.cofactor_smiles = []

	def strip_cofactors(self, coffile="cof.json", pairfile="pairs.json", smfile="small.json"):
		"""Removes cofactors from reactions for fingerprinting and outputs the reduced lists as attributes of the class.
		Requires a file 'spectator_smiles.json' from which to draw its non-cofactor compounds.  Requires consistency.

				kwargs: coffile - file with all cofactors, defaulted to cof.json
						pairfile - file with all cofactor pairs, defaulted to pairs.json
						smfile - file wth small cofactor names, defaulted to small.json"""

		# Load dictionaries.

		f = open(os.path.join(CHEMPATH, coffile))
		cof = json.load(f)
		f.close()

		g = open(os.path.join(CHEMPATH, pairfile))
		pair = json.load(g)
		g.close()

		h = open(os.path.join(CHEMPATH, smfile))
		small = json.load(h)
		h.close()

		brenda_cof = [x for x in self.cofactor_smiles if x not in cof.keys() and x not in small.values()]

		#Check for any of the paired cofactors.

		rxt_clean = [x for x in self.reactant_smiles]
		prod_clean = [x for x in self.product_smiles]
		s = [x for x in self.stoichiometry]

		if all([x in cof.keys() for x in rxt_clean + prod_clean]):
			pass

		else:
			for i, r in enumerate(self.reactant_smiles):
				if r not in cof.keys() and r not in small.values() and r not in brenda_cof:
					pass
				elif r in cof.keys():
					conj = pair[cof[r]]
					for j, p in enumerate(self.product_smiles):
						if p in conj:
							rxt_clean.remove(r)
							prod_clean.remove(p)
							s.remove(self.stoichiometry[i])
							s.remove(self.stoichiometry[j + len(self.reactant_smiles)])
						else:
							pass
				elif r in small.values():
					rxt_clean.remove(r)
					s.remove(self.stoichiometry[i])
				elif r in brenda_cof:
					rxt_clean.remove(r)
					s.remove(self.stoichiometry[i])
				else:
					pass

			for j, p in enumerate(self.product_smiles):
				if p not in small.values() and p not in brenda_cof:
					pass
				elif p in small.values() and p in prod_clean:
					prod_clean.remove(p)
					s.remove(self.stoichiometry[j + len(self.reactant_smiles)])
				elif p in brenda_cof and p in prod_clean:
					prod_clean.remove(p)
					s.remove(self.stoichiometry[j + len(self.reactant_smiles)])
				else:
					pass

		self.reactants = rxt_clean
		self.products = prod_clean
		self.stoich = s

		self.r_fpt = [np.array(fptr.integer_fp(str(x))) for x in self.reactants]
		self.p_fpt = [np.array(fptr.integer_fp(str(x))) for x in self.products]

	def ez_cof_strip(self, coffile="cof.json", smfile="small.json"):
		"""Removes cofactors in a less rigorous way.  Does not rely on paired cofactors, and strips any compound
		that is present in coffile and smfile indiscriminately. This allows for entries that have demonstrated
		substrate data, but no confirmed product data to be used in a given analysis."""

		f = open(os.path.join(CHEMPATH, coffile))
		cof = json.load(f)
		f.close()

		g = open(os.path.join(CHEMPATH, smfile))
		small = json.load(g)
		g.close()

		rxt_clean = [x for x in self.reactant_smiles]
		prod_clean = [x for x in self.product_smiles]
		s = [x for x in self.stoichiometry]

		for i, r in enumerate(self.reactant_smiles):
			if r not in cof.keys() and r not in small.values():
				pass
			elif r in cof.keys():
				rxt_clean.remove(r)
				s.remove(self.stoichiometry[i])
			elif r in small.values():
				rxt_clean.remove(r)
				s.remove(self.stoichiometry[i])
			elif r == "" or r == "?":
				rxt_clean.remove(r)
				s.remove(self.stoichiometry[i])
			else:
				pass

		for j, p in enumerate(self.product_smiles):
			if p not in small.values() and p not in cof.keys():
				pass
			elif p in small.values():
				prod_clean.remove(p)
				s.remove(self.stoichiometry[j + len(self.reactant_smiles)])
			elif p in cof.keys():
				prod_clean.remove(p)
				s.remove(self.stoichiometry[j + len(self.reactant_smiles)])
			elif r == "" or r == "?":
				prod_clean.remove(p)
				s.remove(self.stoichiometry[j + len(self.reactant_smiles)])
			else:
				pass


class SDFReaction(Reaction):

	def __init__(self, mol, assay):

		super(SDFReaction, self).__init__()

		self.reactant_names = mol.data['r_names']
		self.product_names = mol.data['p_names']
		self.reactant_smiles = mol.data['r_smiles']
		self.product_smiles = mol.data['p_smiles']
		self.stoichiometry = mol.data['s']

		self.reactants = [r for r in self.reactant_smiles]
		self.products = [p for p in self.product_smiles]
		self.stoich = [s for s in self.stoichiometry]
		self.r_fpt = [np.array(fptr.integer_fp(str(x))) for x in self.reactants]
		self.p_fpt = [np.array(fptr.integer_fp(str(x))) for x in self.products]

		self.y = int(mol.data[assay])



if __name__ == "__main__":

	res = dbq.pull_docs("Escherichia coli K12", "2.2.1.9")
