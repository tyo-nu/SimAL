__author__ = 'Dante'

from structures.isozyme import BrendaIsozyme
from structures import fingerprinter as fptr

mend = BrendaIsozyme("Escherichia coli K12", "2.2.1.9")


def test_mend_brenda_isozyme_load():
	assert len(mend.rxt_groups) == 2 and len(mend.rxt_groups[0]) == 21 and len(mend.rxt_groups[1]) == 3


def test_mend_analysis():
	mend.analyze_reactions()
	print mend.q
	assert mend.q[0] == ['[#6v4]-!@[#6v4]=!@[#8v2]'] and mend.q[1] == ['[#8v2]-!@[#6v4](-!@[#6v4](-!@[#6v4])=!@[#8v2])=!@[#8v2]']

def test_fp_agreement():
	for smi, fp in mend.pos[0]:
		assert fptr.integer_fp(str(smi)) == fp




