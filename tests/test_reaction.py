from databases import db_queries as dbq

__author__ = 'Dante'

from structures.reaction import BrendaReaction
import hashlib

mend = 'c5d412aa1ba627cc3bef659411e5ed0eb85dd28e'
phedh = '503f26bc1a44217dc90ea7281ce01f03fe95f58a'

mend_fpts = 'f25abfd6771db7fab514ac7dd008179030dd0799'  # ring positions in, chiral flags
phedh_fpts = '689bcfe03f43c6c7bf83aed150efb303df108a74'  # ring positions in, chiral flags

res_mend = dbq.pull_docs('Escherichia coli K12', '2.2.1.9')
res_phedh = dbq.pull_docs('Lysinibacillus sphaericus', '1.4.1.20')


def test_query_mend():
	assert hashlib.sha1(str(res_mend)).hexdigest() == mend


def test_is_consistent_mend():
	assert BrendaReaction(res_mend[0]).is_consistent()


def test_strip_cofactors_mend():
	r = BrendaReaction(res_mend[0])
	r.strip_cofactors()
	assert r.products == [u'OC(=O)CCC(=O)C1=CC[C@@H]([C@H]([C@@H]1C(=O)O)O)OC(=C)C(=O)O'] and r.reactants == [u'C=C(C(=O)O)O[C@H]1C=CC=C([C@@H]1O)C(=O)O', u'OC(=O)CCC(=O)C(=O)O']


def test_fpt_all_mend():
	r = BrendaReaction(res_mend[0])
	r.strip_cofactors()
	assert hashlib.sha1(str(r.fpt_all_())).hexdigest() == mend_fpts


def test_query_phedh():
	assert hashlib.sha1(str(res_phedh)).hexdigest() == phedh


def test_is_consistent_phedh():
	assert BrendaReaction(res_phedh[0]).is_consistent()


def test_strip_cofactors_phedh():
	r = BrendaReaction(res_phedh[0])
	r.strip_cofactors()
	assert r.products == [u'N[C@H](C(=O)O)Cc1ccccc1'] and r.reactants == [u'O=C(C(=O)O)Cc1ccccc1']


def test_fpt_all_phedh():
	r = BrendaReaction(res_phedh[0])
	r.strip_cofactors()
	assert hashlib.sha1(str(r.fpt_all_())).hexdigest() == phedh_fpts
