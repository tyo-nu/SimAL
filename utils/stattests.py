__author__ = 'Dante'

import rpy2.robjects as ro
import numpy as np

def barnard_exact(narray):

	f_array = np.ravel(narray)

	out1 = ro.r('library(Barnard, lib.loc="C:/Users/Dante/Documents/Dante/R/win-library/3.2")')
	out2 = ro.r('capture.output(p <- barnard.test(%s, %s, %s, %s)$p.value)' % (str(int(f_array[0])), str(int(f_array[1])), str(int(f_array[2])), str(int(f_array[3]))))
	p = ro.r('p')[1]  # The p-values are located in the 8th entry of the dataframe as a list: [One-sided, two-sided]

	return p