__author__ = 'Dante'

from databases import db_queries as dbq
from ..structures.reaction import BrendaReaction
import json
import pybel
from ..chem.chem import id_chstring
import os.path


PATH = os.path.dirname(os.path.abspath(__file__))


def mmodel(org, filename_out=False):

	model = os.path.join(PATH, "%s_mod.json") % org

	print model

	if os.path.isfile(model):
		pass

	else:
		org_info = dbq.one_org_rxns(org)
		ec_substr = {}

		for k, v in org_info.iteritems():
			substr = []
			for rxn in v:
				r = BrendaReaction(rxn)
				print r.reactant_smiles
				try:
					if r.is_consistent():
						r.strip_cofactors()
					else:
						r.ez_cof_strip()
					for s in r.reactants:
						substr.append(s)
				except ValueError:
					continue

				ec_substr[k] = list(set(substr))

		with open(org + "_mod.json", 'w') as out:
			out.write(json.dumps(ec_substr, indent=4))

	if filename_out:
		return "%s_mod.json" % org


def simzyme_filter(sz_out, excludable):

	name = sz_out.split('.')[0] + ".sdf"

	f = open(sz_out, 'r')
	contents = f.readlines()
	f.close()

	nat_sub = contents[0].strip('\n\r')
	sz = [(x.split('\t')[0], x.split('\t')[1], x.split('\t')[2], x.split('\t')[3].strip('\n\r')) for x in contents if x.split('\t')[0][0] == '1' or x.split('\t')[0][0] == '0']

	g = open(excludable)
	ex = {k.encode('utf-8'): [v.encode('utf-8') for v in l] for k, l in json.load(g).iteritems()}
	g.close()

	out = pybel.Outputfile("sdf", name, overwrite=True)

	for tc, chstr, chname, ec in sz:
		if ec not in ex.keys():
			mol = pybel.readstring(id_chstring(chstr), chstr)
			mol.data["tc"] = tc
			mol.data["ec_link"] = ec
			mol.data["flags"] = 0
			out.write(mol)
		elif ec in ex.keys() and nat_sub not in ex[ec]:
			mol = pybel.readstring(id_chstring(chstr), chstr)
			mol.data["tc"] = tc
			mol.data["ec_link"] = ec
			mol.data["flags"] = 1
			out.write(mol)
		else:
			continue

	out.close()

if __name__ == "__main__":

	mmodel("Pseudomonas putida")