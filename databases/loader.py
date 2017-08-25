__author__ = 'Dante'

import parser as bp
from pymongo import MongoClient
import pymongo
import json
import pybel
import os
import hashlib
import time
from structures import fingerprinter as fptr
from rdkit.Chem import inchi as rdki

DBPATH = os.curdir
CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")

def get_info(ec, raw):
	"""Get raw info from data dump."""

	try:
		organisms = bp.organism_pull(raw)

		reactions = bp.screen_rxn({ec: raw})

		cofactors = bp.cof_infos(raw)

		return ec, organisms, reactions, cofactors

	except IndexError:

		print ec + "...failed"


def _dbize(ec, org, rxn, cof, all_smiles):
	"""Place data into MongoDB."""

	#Connect to mongodb.
	client = MongoClient()
	db = client.BrendaDB
	ec_collection = db.ec_pages
	rxn_collection = db.rxn_pages
	cpd_collection = db.cpd_pages

	#Build dictionary of reactions and organisms

	r_o_dict = {}

	for k, v in rxn.iteritems():
		p_ = []
		r_ = []
		#The substrates/products are in name format from the dump, so adding smiles data here.
		if len(v[1]) > 0 and len(v[2]) > 0:
			for comp in v[1]:
				if comp in all_smiles:
					smiles = all_smiles[str(comp)]
					id = hashlib.sha1(smiles).hexdigest()
					inchi = pybel.readstring('smi', smiles).write('inchi').strip('\t\n')
					inchikey = rdki.InchiToInchiKey(inchi)
					r_.append(id)
					cpd_collection.update({"_id": id}, {"$set": {"smiles": smiles, "inchi": inchi, "inchikey": inchikey, "name": comp}}, upsert=True)
				else:
					r_.append('')
			for comp in v[2]:
				if comp in all_smiles:
					smiles = all_smiles[str(comp)]
					id = hashlib.sha1(smiles).hexdigest()
					inchi = pybel.readstring('smi', smiles).write('inchi').strip('\t\n')
					inchikey = rdki.InchiToInchiKey(inchi)
					p_.append(id)
					cpd_collection.update({"_id": id}, {"$set": {"smiles": smiles, "inchi": inchi, "inchikey": inchikey, "name": comp}}, upsert=True)
				else:
					p_.append('')
			#A reaction doc is generated containing the names/smiles of both products and reactants as well as a
			#stoichiometry vector. The id field is a hash of the final dictionary, and gets added into the rxn/org dict
			#for inclusion in the ec pages.  Upsert option adds to anything that matches the query and creates a new
			#entry if there is no match.
			r_entry = {"r_name": v[1], "p_name": v[2], "r_smiles": r_, "p_smiles": p_, "s": v[3]}
			rxn_collection.update({"_id": hashlib.sha1(str(r_entry)).hexdigest()}, {"$set": {"rxn": r_entry}}, upsert=True)
			r_o_dict[k] = (v[0], hashlib.sha1(str(r_entry)).hexdigest())
		else:
			continue

	#Iterate through a dictionary of organisms to create the ec pages. Each doc is for a particular organism and lists
	#all of the ecs present in it, followed by a list of reactions in each ec listing, with cofactors.
	for k, v in org.iteritems():
		rxns_in = [x[1] for x in r_o_dict.values() if k in x[0]]
		cofs_in = [{"name": x[1], "link": ''} for x in cof if k in x[0]]
		for d in cofs_in:
			if d["name"] in all_smiles:
				d["link"] = hashlib.sha1(all_smiles[str(d["name"])]).hexdigest()
			else:
				d["link"] = ''
		ec_collection.update({"org": v}, {"$set": {"ec." + ec.replace('.', '_'): {"rxns": rxns_in, "cofactors": cofs_in}}}, upsert=True)


def dbize_zinc(zincfolder):
	"""Move zinc dumps into MongoDB. Takes as an argument the folder in which zinc sdfs are stored and ignores
	all non-sdf files for convenience."""

	sdffiles = [zincfolder + "/" + x for x in os.listdir(zincfolder) if x[-4:] == '.sdf']

	#Connect to mongodb.
	client = MongoClient()
	db = client.zincDB
	collection = db.zinc_pages

	#Iterate through sdf files.
	for sdfin in sdffiles:
		f = pybel.readfile('sdf', sdfin)

		#Populate zinc docs with smiles and pertinent info. Id field is hashed ZINCID.
		for mol in f:
			smiles = mol.write('can').split('\t')[0]
			fpt = fptr.integer_fp(smiles)
			zinc_doc = {'_id': mol.write('smi').split('\t')[1].strip('\n'), 'smiles': smiles, 'fp': fpt, "mw": mol.exactmass, "atoms": len(mol.atoms), "formula": mol.formula}
			collection.save(zinc_doc)

		f.close()


def apply_vendors(xlfile):
	client = MongoClient()
	db = client.zincDB
	cpd = db.zinc_pages

	start = time.time()

	n = 0

	with open(xlfile, 'r') as g:
		for line in g:
			n += 1
			if n % 2200 == 0:
				stop = time.time()
				print "Time elapsed: %s s." % str(stop - start)
			sl = line.split('\t')
			cpd.update({"_id": sl[0]}, {"$push": {"vendors": {sl[1].replace(u'.', u''): sl[2]}}})


def brenda_up(loc=DBPATH, smiles_dump_brenda='BrendaNorm.json', smiles_dump_bkm='BKMNorm.json'):

	# Import cheminformatic dumps from json.
	f = open(os.path.join(CHEMPATH, smiles_dump_brenda))
	brenda_smiles = {k.encode('utf-8'): v.encode('utf-8') for k, v in json.load(f).iteritems()}
	f.close()

	g = open(os.path.join(CHEMPATH, smiles_dump_bkm))
	bkm_smiles = {k.encode('utf-8'): v.encode('utf-8') for k, v in json.load(g).iteritems() if v.encode('utf-8') != ""}
	g.close()

	all_smiles = {k: pybel.readstring('smi', v).write('can').strip('\t\n') for k, v in dict(brenda_smiles.items() + bkm_smiles.items()).iteritems()}

	i = bp.getECInfo(location=loc)

	for k, v in i.iteritems():
		e, o, r, c = get_info(k, v)
		_dbize(e, o, r, c, all_smiles)

	client = MongoClient()
	db = client.BrendaDB
	ec_collection = db.ec_pages

	ec_collection.ensure_index([("org", pymongo.TEXT)])





