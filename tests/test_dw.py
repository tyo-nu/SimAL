__author__ = 'Dante'

from scripts import density_weight as dw
import math
import random


def test_entropy():

	p = random.random()

	assert round(dw.entropy(p), 3) == round(-(p * math.log(p, 2) + (1 - p) * math.log((1 - p), 2)), 3)


def test_entropy_max():

	assert round(dw.entropy(0.5), 2) == 1.0


def test_entropy_commute():

	p = round(random.random(), 3)

	assert round(dw.entropy(p), 3) == round(dw.entropy(1 - p), 3)
