__author__ = 'Dante'

import numpy as np
import math
import json
import matplotlib as plt
from chem import chem
from matplotlib import pyplot
import scipy.cluster.hierarchy as sch
from structures import fingerprinter as fptr


def dw_rand_curves(n, rand_res, dw_res, saveas, iterations=1000, c=1.0, decf=False):

	rand_avg = np.sum(np.vstack(tuple(rand_res)), 0) / iterations
	dw_avg = np.sum(np.vstack(tuple(dw_res)), 0) / iterations
	rand_err = 1.96 * np.std(np.vstack(tuple(rand_res)), 0) / math.sqrt(iterations)
	dw_err = 1.96 * np.std(np.vstack(tuple(dw_res)), 0) / math.sqrt(iterations)
	xdesc = 'Number of Training Samples'
	ydesc = 'Accuracy Score'
	plt.rcParams['font.sans-serif'] = ['Arial']

	pyplot.errorbar(n, rand_avg, fmt='s-', yerr=rand_err, color='darkred', markersize=9, lw=2,
	                label='Random Selection')
	pyplot.errorbar(n, dw_avg, fmt='v-', yerr=dw_err, color='darkblue', markersize=9, lw=2,
	                label='Active Learning')
	pyplot.tick_params(labelsize=14)
	leg_title = "Final Random Accuracy = %s\nFinal DW Accuracy = %s\nC = %s" % (
		str(round(rand_avg[-1], 3) * 100) + '%', str(round(dw_avg[-1], 3) * 100) + '%', str(c))
	pyplot.legend(loc=4, title=leg_title)
	pyplot.xlabel(xdesc, size=18, labelpad=14)
	pyplot.ylabel(ydesc, size=18, labelpad=14)
	pyplot.savefig(saveas + "C_%s_decisionf_%s.svg" % (str(c), str(decf)))
	print "Random: Mu -- %.4f Sigma -- %.4f DW: Mu -- %.4f Sigma -- %.4f" % (rand_avg[-1], rand_err[-1], dw_avg[-1], dw_err[-1])
	# pyplot.show()


def dw_heatmap(positions, smiles_acc, data_in_, classes, saveas, c=1.0, decf=False, yl='Position Selected', zl='Selection Frequency'):
	plt.rcParams['font.sans-serif'] = ['Arial']
	fig = pyplot.figure()
	# Plots a dendrogram above the heatmap.
	axdendro = fig.add_axes([0.06, 0.68, 0.8, 0.27])

	fpts = np.vstack(tuple([fptr.integer_fp(str(s)) for s in smiles_acc]))
	Y = sch.linkage(fpts, method='single', metric='jaccard')
	Z = sch.dendrogram(Y, orientation='top')
	axdendro.set_xticks([])
	axdendro.set_yticks([])

	#Plotting the heat map.  'pcolor' outputs a mappable object that is used as a mandatory argument to 'colorbar().'
	#add axes arguments are: distance from left, distance from bottom, width, height.

	ax = fig.add_axes([0.06, 0.05, 0.8, 0.6])
	#Grab the order of the leaves of the dendrogram so the heatmap can be reordered to match.
	index = Z['leaves']
	D = data_in_.T[index]
	hmap = ax.pcolor(D.T, cmap='gist_heat')

	horiz = np.arange(data_in_.shape[1]) + 0.5
	vert = np.arange(data_in_.shape[0]) + 0.5

	pyplot.ylim([0, vert[-1] + 0.5])
	pyplot.xlim([0, horiz[-1] + 0.5])
	pyplot.ylabel(yl, size=16)
	ax.set_xticks(horiz, minor=False)
	ax.set_yticks(vert, minor=False)

	names = []
	for s in classes['smiles']:
		name_entry = chem.calc_name(s)
		names.append(unicode(name_entry, "utf-8"))

	col = [names[m] + ' (%s)' % str(classes['label'][m]) for m in index]
	row = positions

	ax.set_xticklabels(col, minor=False, rotation=90, ha='center', size=9)
	ax.set_yticklabels(row, minor=False, size=11)

	#Plots the colorbar on separate axes so that the dendrogram can be aligned to the heatmap alone.
	axcolor = fig.add_axes([0.89, 0.05, 0.02, 0.6])
	cbar = pyplot.colorbar(hmap, cax=axcolor)
	axcolor.set_ylabel(zl, size=16, rotation=270)
	#Eliminates white lines in Inkscape due to viewer bug; makes colorbar render with overlapping segments.
	cbar.solids.set_edgecolor("face")

	pyplot.savefig(saveas + "C_%s_decisionf_%s.svg" % (str(c), str(decf)))
	pyplot.show()

def similarity_matrix(smiles_acc, data_in_, classes, saveas, yl='Position Selected', zl='Selection Frequency', num_labels=False):

	plt.rcParams['font.sans-serif'] = ['Arial']
	fig = pyplot.figure()
	# Plots a dendrogram above the heatmap.
	axdendro = fig.add_axes([0.02, 0.9, 0.85, 0.1])
	#axdendro = fig.add_axes([0.12, 0.8, 0.75, 0.18])

	fpts = np.vstack(tuple([fptr.integer_fp(str(s)) for s in smiles_acc]))
	Y = sch.linkage(fpts, method='single', metric='jaccard')
	Z = sch.dendrogram(Y, orientation='top')
	axdendro.set_xticks([])
	axdendro.set_yticks([])

	# Plotting the heat map.  'pcolor' outputs a mappable object that is used as a mandatory argument to 'colorbar().'
	#add axes arguments are: distance from left, distance from bottom, width, height.

	ax = fig.add_axes([0.02, 0.05, 0.85, 0.85])
	#ax = fig.add_axes([0.12, 0.05, 0.75, 0.75])
	#Grab the order of the leaves of the dendrogram so the heatmap can be reordered to match.
	index = Z['leaves']
	D = data_in_.T[index].T[index]
	hmap = ax.pcolor(D.T, cmap='gist_heat')

	horiz = np.arange(data_in_.shape[1]) + 0.5
	vert = np.arange(data_in_.shape[0]) + 0.5

	pyplot.ylim([0, vert[-1] + 0.5])
	pyplot.xlim([0, horiz[-1] + 0.5])
	pyplot.ylabel(yl, size=16)
	ax.set_xticks(horiz, minor=False)
	ax.set_yticks(vert, minor=False)

	names = []
	for s in classes['smiles']:
		name_entry = chem.calc_name(s)
		names.append(unicode(name_entry, "utf-8"))

	if num_labels:
		col = [i + 1 for i in range(len(names))]
		with open('tcmap_labels.json', 'w') as outfile:
			outfile.write(json.dumps({k: v for k, v in zip(col, [names[m] for m in index])}, indent=4))
	else:
		col = [names[m] for m in index]

	ax.set_xticklabels(col, minor=False, rotation=0, ha='center', size=7)
	ax.set_yticklabels(col, minor=False, size=7)

	#Plots the colorbar on separate axes so that the dendrogram can be aligned to the heatmap alone.
	axcolor = fig.add_axes([0.89, 0.05, 0.02, 0.9])
	cbar = pyplot.colorbar(hmap, cax=axcolor)
	axcolor.set_ylabel(zl, size=16, rotation=270)
	#Eliminates white lines in Inkscape due to viewer bug; makes colorbar render with overlapping segments.
	cbar.solids.set_edgecolor("face")

	pyplot.savefig(saveas + '.svg')
	pyplot.show()