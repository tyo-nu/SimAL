__author__ = 'Dante'

from ..structures import fingerprinter as fptr
import hashlib

smiles_a = 'OCC1CS[As](S1)c1ccc(cc1)Nc1nc(=N)[nH]c(=N)[nH]1'
smiles_b = 'OC(=O)[C@H](CCOP(=O)(O)O)N'


def test_integer_fp():

	fp = fptr.integer_fp(smiles_a)

	assert hashlib.sha1(str(fp)).hexdigest() == '46589124425c8f61a93c9442e1dee2ec3dbc9bfb'


def test_integer_sim_identity():

	fpa = fptr.integer_fp(smiles_a)
	fpb = fptr.integer_fp(smiles_b)

	assert round(fptr.integer_sim(fpa, fpa), 3) == 1.0
	assert round(fptr.integer_sim(fpb, fpb), 3) == 1.0


def test_integer_sim_commute():

	fpa = fptr.integer_fp(smiles_a)
	fpb = fptr.integer_fp(smiles_b)

	assert round(fptr.integer_sim(fpa, fpb), 3) == 0.208
	assert round(fptr.integer_sim(fpb, fpa), 3) == 0.208


def test_bin_tc_identity():
	fpa = fptr.integer_fp(smiles_a)
	fpb = fptr.integer_fp(smiles_b)

	assert round(fptr.bin_tc(fpa, fpa), 3) == 1.0
	assert round(fptr.bin_tc(fpb, fpb), 3) == 1.0


def test_bin_tc_commute():

	fpa = fptr.integer_fp(smiles_a)
	fpb = fptr.integer_fp(smiles_b)

	assert round(fptr.bin_tc(fpa, fpb), 3) == 0.107
	assert round(fptr.bin_tc(fpb, fpa), 3) == 0.107