__author__ = 'Dante'

from pymongo import MongoClient
import random
import pybel


def pull_docs(org, ec):
	"""Support function for querying BrendaDB.  Returns a list of tuples with (reaction document, list of cofactors docs
	listed from the BRENDA page."""

	q_ec = ec.replace('.', '_')

	client = MongoClient()
	db = client.BrendaDB
	ec_collection = db.ec_pages
	rxn_collection = db.rxn_pages
	cpd_collection = db.cpd_pages

	result = ec_collection.find({"ec." + q_ec: {"$exists": True}, "$text": {"$search": org}})

	results = []

	for page in result:
		for rxn_id in page["ec"][q_ec]["rxns"]:
			r = rxn_collection.find_one({"_id": rxn_id})["rxn"]

			p_ = []
			for h in r["p_smiles"]:
				if h != "":
					p_.append(cpd_collection.find_one({"_id": h})["smiles"])
				else:
					p_.append("")

			r["p_smiles"] = p_

			r_ = []
			for h in r["r_smiles"]:
				if h != "":
					r_.append(cpd_collection.find_one({"_id": h})["smiles"])
				else:
					r_.append("")

			r["r_smiles"] = r_

			cof_part = page["ec"][q_ec]["cofactors"]

			for c in cof_part:
				try:
					s = cpd_collection.find_one({"_id": c["link"]})["smiles"]
					c["smiles"] = s
				except TypeError:
					c["smiles"] = []

			t = (r, cof_part)
			results.append(t)

	return results


def ec_in(org):

	client = MongoClient()
	db = client.BrendaDB
	ec_collection = db.ec_pages

	res = ec_collection.find({"$text": {"$search": org}})

	for y in [str(x).replace('_', '.') for page in res for x in page["ec"].keys()]:
		yield y


def one_org_rxns(org):

	client = MongoClient()
	db = client.BrendaDB
	ec_collection = db.ec_pages
	rxn_collection = db.rxn_pages
	cpd_collection = db.cpd_pages

	result = ec_collection.find({"$text": {"$search": org}})

	ec_results = []
	rxn_results = []

	for page in result:
		for k in page["ec"].keys():
			ec_results.append(str(k.replace(u'_', u'.')))
		for r_dict in page["ec"].values():
			r = []
			for rxn_id in r_dict["rxns"]:
				rx = rxn_collection.find_one({"_id": rxn_id})["rxn"]

				p_ = []
				for h in rx["p_smiles"]:
					if h != "":
						p_.append(cpd_collection.find_one({"_id": h})["smiles"])
					else:
						p_.append("")

				rx["p_smiles"] = p_

				r_ = []
				for h in rx["r_smiles"]:
					if h != "":
						r_.append(cpd_collection.find_one({"_id": h})["smiles"])
					else:
						r_.append("")

				rx["r_smiles"] = r_

				r.append((rx, {}))

			rxn_results.append(r)

	out = {k: [] for k in ec_results}

	for e, r in zip(ec_results, rxn_results):
		out[e] += r

	return out


def zinc_pull(query, mw_avg, mw_std, zinc_tol_l=1, zinc_tol_r=1):

	client = MongoClient()
	db = client.zincDB
	zinc = db.zinc_pages

	try:
		and_query = query.split(';')[0].split(',')
	except IndexError:
		and_query = []
	try:
		or_query = query.split(';')[1].split(',')
	except IndexError:
		or_query = []

	q = {"fp." + str(int(x) - 1): {"$gt": 0} for x in and_query if x is not ''}
	if len(or_query) > 1:
		q["$or"] = [{"fp." + str(int(x) - 1): {"$gt": 0}} for x in or_query if x is not '']
	q["mw"] = {"$gt": mw_avg - zinc_tol_l * mw_std, "$lt": mw_avg + zinc_tol_r * mw_std}

	mass_results = zinc.find(q)

	return [page for page in mass_results]


def kegg_pull(query):

	client = MongoClient()
	db = client.KEGG
	cpds = db.compounds

	try:
		and_query = query.split(';')[0].split(',')
	except IndexError:
		and_query = []
	try:
		or_query = query.split(';')[1].split(',')
	except IndexError:
		or_query = []

	q = {"FP4": int(x) for x in and_query}
	if len(or_query) > 1:
		q["$or"] = [{"FP4": int(y)} for y in or_query]

	mass_results = cpds.find(q)

	return [page for page in mass_results]


def kegg_xref(keggfile):

	client = MongoClient()
	db = client.zincDB
	zinc = db.zinc_pages

	kegg_res = pybel.readfile('sdf', keggfile)
	outfile = pybel.Outputfile('sdf', keggfile[:-4] + '_xref.sdf')

	for mol in kegg_res:
		inchik = mol.write('inchikey').strip() + '\n'
		if zinc.find({"inchikey": inchik}).count() > 0:
			outfile.write(mol)


def rand_kegg(n):

	client = MongoClient()
	db = client.KEGG
	cpds = db.compounds

	poss_kegg = ['C' + str(x + 1).zfill(5) for x in range(16341)]

	smiles = {}

	while len(smiles.keys()) < n:

		k = random.sample(poss_kegg, 1)[0]
		res = cpds.find({"DB_links.KEGG": k})

		if res.count() == 0:
			continue
		if u'R' in res[0]["SMILES"]:
			continue

		smiles[k] = str(res[0]["SMILES"])

	return smiles.values()

def rand_seed(n):

	client = MongoClient()
	db = client.modelSEED
	cpds = db.compounds

	seeds = []

	while len(seeds) < n:

		k = random.sample(range(12694), 1)[0] + 1
		res = cpds.find({"random": k})

		if res.count() == 0 or u'*' in res[0]['smiles']:
			continue
		else:
			seeds.append(str(res[0]['smiles']))

	return seeds

def brenda_cpds():

	client = MongoClient()
	db = client.BrendaDB
	cpd_collection = db.cpd_pages

	return cpd_collection.find({})

if __name__ == "__main__":

	kegg_pull('84')
