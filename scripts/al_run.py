__author__ = 'Dante'

import os
import math
import machines
import density_weight as dw
from structures.isozyme import BrendaIsozyme as bi
from databases import db_queries as dbq
from structures import fingerprinter as fptr
import numpy as np
import routines
import pybel

CHEMPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chem")

def al_run(org, ec, neg, k, beta=1, pos=None, ent=False, kernel='rbf', degree=3, zinc=True, zinc_tol_l=1, zinc_tol_r=1, greedy=False, vl=None, simfp=fptr.integer_sim, C=5, target_bits=None, screen=None):

	#Collects isozyme data into the Isozyme class.
	a = bi(org, ec)
	if pos:
		a.add_from_sdf(pos, k, pos=True)
	a.add_from_sdf(neg, k, pos=False)

	#Two branches here; one pulls potential test data from ZINC, another pulls from KEGG.

	if zinc:
		res_ = [(page["smiles"], fptr.integer_fp(str(page["smiles"])), page["vendors"], page["_id"]) for page in dbq.zinc_pull(target_bits, a.mass_avg[k], a.mass_std[k], zinc_tol_l=zinc_tol_l, zinc_tol_r=zinc_tol_r) if u'R' not in page["smiles"] and 'vendors' in page]
		res_s = [rr for rr in res_ if rr[1] is not None]
		if screen is not None:
			patt = [pybel.Smarts(smarts) for smarts in screen.split('|')]
			if len(patt) > 2:
				raise IOError('al_run only supports OR filters for two SMARTS queries at this time.')
			res = [rr for rr in res_s if len(patt[0].findall(pybel.readstring('smi', str(rr[0])))) > 0 or len(patt[1].findall(pybel.readstring('smi', str(rr[0])))) > 0]
		else:
			res = res_s

	else:

		res = [(page["SMILES"], np.array(fptr.integer_fp(str(page["SMILES"])))) for page in dbq.kegg_pull(target_bits) if u'R' not in page["SMILES"] and np.array(fptr.integer_fp(str(page["SMILES"]))) is not None]

	labels = machines.svm_clf(a.pos[k], a.neg[k], res, kernel=kernel, degree=degree, ent=ent, C=C)

	test_a = np.vstack(tuple([np.array(x[1]) for x in res if x[1] is not None and len(x[1]) == 313]))

	tc_u = dw.avg_proximity(test_a, test_a, f=simfp)

	if greedy:

		if ent:

			xis = [l * dw.weight(dw.entropy(p), tc_u[i], beta=beta) for i, (l, p) in enumerate(labels)]

		else:

			xis = [l * dw.weight(dw.hyper_distance(d), tc_u[i], beta=beta) for i, (l, d) in enumerate(labels)]

	else:

		if ent:

			xis = [dw.weight(dw.entropy(p), tc_u[i], beta=beta) for i, (l, p) in enumerate(labels)]

		else:

			xis = [dw.weight(dw.hyper_distance(d), tc_u[i], beta=beta) for i, (l, d) in enumerate(labels)]

	if zinc:
		dw.generate_report(sorted(zip([s for s, fp, vend, z in res if fp is not None], xis, [lab[0] for lab in labels], [vend for s, fp, vend, z in res if fp is not None], [z for s, fp, vend, z in res if fp is not None]), key=lambda y: y[1], reverse=True), vendors_list=vl, outfile="%s_ec%s_beta%s_%s_zinc%s%s_C%s.sdf" % (org, ec.replace('.', '_'), str(beta), kernel, str(zinc_tol_l).replace('.', '_'), str(zinc_tol_r).replace('.', '_'), str(C)))
		f = open("%s_ec%s_beta%s_%s_zinc%s%s_C%s.txt" % (org, ec.replace('.', '_'), str(beta), kernel, str(zinc_tol_l).replace('.', '_'), str(zinc_tol_r).replace('.', '_'), str(C)), 'w')

	else:
		dw.generate_report(sorted(zip([s for s, fp in res], xis, [lab[0] for lab in labels]), key=lambda y: y[1], reverse=True), outfile="%s_ec%s_beta%s_%s.sdf" % (org, ec.replace('.', '_'), str(beta), kernel), zinc=False)
		f = open("%s_ec%s_beta%s_%s.txt" % (org, ec.replace('.', '_'), str(beta), kernel), 'w')

	for score in xis:
		f.write(str(score) + '\n')
	f.close()

def dissim_run(org, ec, neg, k, pos=None, zinc=False, zinc_tol_l=1, zinc_tol_r=1, vl=None, simfp=fptr.integer_sim, target_bits=None, screen=None):

		# Collects isozyme data into the Isozyme class.
		a = bi(org, ec)
		bits = a.analyze_reactions()
		if pos:
			a.add_from_sdf(pos, k, pos=True)

		a.add_from_sdf(neg, k, pos=False)
		#Two branches here; one pulls potential test data from ZINC, another pulls from KEGG.

		res_ = [(page["smiles"], fptr.integer_fp(str(page["smiles"])), page["vendors"], page["_id"]) for page in dbq.zinc_pull(target_bits, a.mass_avg[k], a.mass_std[k], zinc_tol_l=zinc_tol_l, zinc_tol_r=zinc_tol_r) if u'R' not in page["smiles"] and 'vendors' in page]
		res_s = [rr for rr in res_ if rr[1] is not None]
		if screen is not None:
			patt = [pybel.Smarts(smarts) for smarts in screen.split('|')]
			if len(patt) > 2:
				raise IOError('al_run only supports OR filters for two SMARTS queries at this time.')
			res = [rr for rr in res_s if len(patt[0].findall(pybel.readstring('smi', str(rr[0])))) > 0 or len(patt[1].findall(pybel.readstring('smi', str(rr[0])))) > 0]

		else:
			res = res_s

		x_pos_array = np.vstack(tuple([t[1] for t in a.pos[k]]))
		x_neg_array = np.vstack(tuple([t[1] for t in a.neg[k]]))
		x_array = np.vstack((x_pos_array, x_neg_array))
		centroid = np.mean(x_array, axis=0)

		test_a = np.vstack(tuple([np.array(x[1]) for x in res if x[1] is not None]))
		test_centroid = np.mean(test_a, axis=0)
		tc_u = dw.avg_proximity(test_a, test_a, f=simfp)

		xis_a = [(x[0], fptr.integer_sim(centroid, x[1]), 1, x[2], x[3]) for x in res if x[1] is not None]
		xis_b = [(x[0], tc_u[i] * (-math.log(fptr.integer_sim(centroid, x[1]), 2)), 1, x[2], x[3]) for i, x in enumerate(res) if x[1] is not None]

		dw.generate_report(sorted(xis_a, key=lambda y: y[1]), vendors_list=vl, outfile="%s_ec%s_dissim_zinc%s%s.sdf" % (org, ec.replace('.', '_'), str(zinc_tol_l).replace('.', '_'), str(zinc_tol_r).replace('.', '_')))
		dw.generate_report(sorted(xis_b, key=lambda y: y[1]), vendors_list=vl, outfile="%s_ec%s_dissimcentral_zinc%s%s.sdf" % (org, ec.replace('.', '_'), str(zinc_tol_l).replace('.', '_'), str(zinc_tol_r).replace('.', '_')))


def al_xval_ins(org, ec, k, neg=None, pos=None, beta=1.0, kernel='rbf', gamma=0.005, iterations=100, batch=1, C=1.0, initial=2, decf=True, simfp=fptr.integer_sim):
	a = bi(org, ec)
	if neg is not None:
		if pos:
			a.add_from_sdf(pos, k, pos=True)
		a.add_from_sdf(neg, k, pos=False)
		a.xval_selection(k, beta=beta, batch=batch, kernel=kernel, iterations=iterations, initial=initial, c=C, gamma=gamma, decf=decf, simfp=simfp)
	else:
		a.xval_selection_random(k, beta=beta, batch=batch, kernel=kernel, iterations=iterations, initial=initial, c=C, gamma=gamma, decf=decf, simfp=simfp)


def al_exp_val(org, ec, k, exp, neg=None, beta=1.0, kernel='rbf', degree=3, gamma=0.005, iterations=100, batch=1, C=1.0, initial=2, decf=False, random_seed=None, pos=None, simfp=fptr.integer_sim):

	a = bi(org, ec)
	if neg is not None:
		if pos:
			a.add_from_sdf(pos, k, pos=True)
		a.add_from_sdf(neg, k, pos=False)
		suppl = pybel.readfile('sdf', os.path.join(CHEMPATH, exp))
		excl = []
		for mol in suppl:
			smi = mol.write('can').strip()
			cls = int(mol.data['label'])
			a.add_from_smiles(smi, k, cls)
			excl.append(smi)
		a.expval_selection(k, excl, c=C, gamma=gamma, iterations=iterations, batch=batch, degree=degree, kernel=kernel, beta=beta, decf=decf, seed=random_seed, simfp=simfp, initial=initial)
	else:
		a.expval_selection_random(k, exp, c=C, gamma=gamma, iterations=iterations, batch=batch, degree=degree, kernel=kernel, beta=beta, decf=decf, seed=random_seed, simfp=simfp, initial=initial)


def al_exp_ins(org, ec, k, exp, neg=None, beta=1.0, kernel='rbf', degree=3, gamma=0.005, iterations=100, batch=1, C=1.0, initial=2, decf=False, random_seed=None, fp='FP4', simfp=fptr.integer_sim):

	a = bi(org, ec)
	if neg is not None:
		a.add_from_sdf(neg, k, pos=False)
	else:
		a.random_negatives(k)

	suppl = pybel.readfile('sdf', os.path.join(CHEMPATH, exp))
	excl = []
	for mol in suppl:
		smi = mol.write('can').strip()
		cls = int(mol.data['label'])
		a.add_from_smiles(smi, k, cls)
		excl.append(smi)

	smiles_access = [t[0] for t in a.pos[k]] + [t[0] for t in a.neg[k]]
	n = max([len(str(x)) for x in smiles_access])

	if fp == 'FP4':
		x_pos_array = np.vstack(tuple([t[1] for t in a.pos[k]]))
		x_neg_array = np.vstack(tuple([t[1] for t in a.neg[k]]))

		y_obj = []

		y_obj += [1] * x_pos_array.shape[0]
		y_obj += [-1] * x_neg_array.shape[0]

		x = np.vstack((x_pos_array, x_neg_array))
		y = np.array(zip(y_obj, smiles_access), dtype=[('label', 'i4'), ('smiles', '|S%s' % str(n))])

	elif fp == 'FP2':
		x_pos_array = np.vstack(tuple([np.array(fptr.reconstruct_fp(t[0], fptype='FP2')) for t in a.pos[k]]))
		x_neg_array = np.vstack(tuple([np.array(fptr.reconstruct_fp(t[0], fptype='FP2')) for t in a.neg[k]]))

		y_obj = []

		y_obj += [1] * x_pos_array.shape[0]
		y_obj += [-1] * x_neg_array.shape[0]

		x = np.vstack((x_pos_array, x_neg_array))
		y = np.array(zip(y_obj, smiles_access), dtype=[('label', 'i4'), ('smiles', '|S%s' % str(n))])

	else:
		raise IOError("Valid values for fp are FP2 and FP4.")

	outfile = "al_expins_%s_%s_beta%s_batch%s_%s_rseed%s" % (org, ec, str(beta).replace('.', ''), str(batch), kernel, str(random_seed))

	out = routines.dw_exp_ins(x, y, outfile, smiles_access, excl, C=C, gamma=gamma, iterations=iterations, batch=batch, degree=degree, kernel=kernel, beta=beta, decf=decf, seed=random_seed, simfp=simfp, initial=initial)

if __name__ == "__main__":
	#prog = al_xval("Xanthomonas citri", "3.1.1.43", 0, neg="AAEHNegatives.sdf", C=5, batch=1, beta=1, iterations=1000, total_tc=True, fp='FP4', kernel='rbf', decf=True)
	#al_inspect("rhodochrous", "4.2.1.84", "NHaseNegatives.sdf", 0, C=5, batch=1, beta=1, iterations=1000)
	#al_inspect("Xanthomonas citri", "3.1.1.43", "AAEHNegatives.sdf", 0, C=5, batch=1, beta=1, iterations=1000, decf=True)
	#al_xval_ins("Escherichia coli K12", "2.2.1.9", 0, neg="MenDNegatives.sdf", pos="MenDPositives_extra.sdf", C=5, batch=1, beta=1, iterations=1000, kernel='rbf', decf=True)
	#al_xval_ins("iowensis", "1.2.99.6", 0, neg="CarNegatives.sdf", pos="CarPositives.sdf", C=5, batch=1, beta=1, iterations=1000, kernel='rbf', decf=True)
	#al_xval_ins("putida", "1.14.13.84", 0, neg="HAPMONegatives.sdf", C=5, batch=1, beta=1, iterations=1000, kernel='rbf', decf=True)
	#al_xval_ins("Xanthomonas citri", "3.1.1.43", 0, neg="AAEHNegatives.sdf", C=5, batch=1, beta=1, iterations=1000, kernel='rbf', decf=True)
	#al_inspect("Escherichia coli K12", "2.2.1.9", "MenDNegatives.sdf", 0, C=5, batch=1, beta=1, iterations=1000, decf=True)
	#al_exp_val("Escherichia coli K12", "2.2.1.9", 0, "MenD_rd1.sdf", neg="MenDNegatives_extra.sdf", pos="MenDPositives_extra.sdf", C=5, batch=1, beta=1, iterations=1000, decf=True, kernel='rbf')
	#al_exp_ins("Escherichia coli K12", "2.2.1.9", 0, "MenD_rd1.sdf", neg="MenDNegatives.sdf", C=5, batch=1, beta=1, iterations=1000, decf=True)
	al_run("iowensis", "1.2.99.6", "CarNegatives.sdf", 0, pos="CarPositives.sdf", zinc=False, target_bits='84')