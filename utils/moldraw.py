__author__ = 'Dante'

import os
from chem import chem
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import hashlib
import svgutils.transform as sg

IMAGEPATH = os.path.join(os.path.dirname(__file__), "img")

def _colorizer(mol, query, exc=True):

	atom_list = []

	query_mols = [Chem.MolFromSmarts(qq) for qq in list(query)]

	for x in query_mols:
		matches_ = mol.GetSubstructMatches(x)
		for atoms_ in matches_:
			atom_list += atoms_

	if len(atom_list) != 0:
		return atom_list
	elif len(atom_list) == 0 and exc:
		return None
	else:
		return atom_list

def moldraw(smi, size=(450, 150), kekulize=True, highlight_atoms=None):

	filename = os.path.join(IMAGEPATH, hashlib.sha1(smi).hexdigest() + '.svg')

	mol = Chem.MolFromSmiles(smi)
	mc = Chem.Mol(mol.ToBinary())
	if kekulize:
		try:
			Chem.Kekulize(mc)
		except:
			mc = Chem.Mol(mol.ToBinary())

	if not mc.GetNumConformers():
		rdDepictor.Compute2DCoords(mc)
	if highlight_atoms:
		coord_ = _colorizer(mc, highlight_atoms)
	else:
		coord_ = None

	if coord_ is not None:
		dwr = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
		dwr.DrawMolecule(mc, highlightAtoms=coord_)
		dwr.FinishDrawing()
		svg = dwr.GetDrawingText().replace('svg:', '')
		svg_noframe = '\n'.join([line for line in svg.split('\n') if line[0:5] != '<rect'])

		f = open(filename, 'w')
		f.write(svg_noframe)
		f.close()

		return filename


class MolGrid(object):

	def __init__(self, size=(1200, 3600), rowl=5):

		if not isinstance(size, tuple):
			raise TypeError('Argument size must be a tuple of the form (width, height).')

		self.__smiles = []
		self.__subimages = []
		self.panels = len(self.__smiles)
		self.width = size[0]
		self.height = size[1]
		self.row = rowl

	def add_smiles(self, smi):

		self.__smiles.append(smi)
		self.panels = len(self.__smiles)

	def generate_figures(self, fname, highlight=None):

		_image_len = self.width / self.row
		self.__subimages = [(s, moldraw(s, size=(_image_len, _image_len), highlight_atoms=highlight)) for s in self.__smiles]

		_fig = sg.SVGFigure(str(self.width), str(self.height))
		_subfigs = [sg.fromfile(ff[1]) for ff in self.__subimages if ff[1] is not None]
		_labels = range(len(_subfigs))
		_plots = [ff.getroot() for ff in _subfigs]

		_x_coord = 0
		_y_coord = 0
		_final_coords = []

		for sf in _plots:
			if _x_coord + (self.width % self.row) == self.width:
				_x_coord = 0
				_y_coord += 1.25 * _image_len
				sf.moveto(_x_coord, _y_coord, scale=1.0)
				_final_coords.append((_x_coord, _y_coord))
				_x_coord += _image_len
			else:
				sf.moveto(_x_coord, _y_coord, scale=1.0)
				_final_coords.append((_x_coord, _y_coord))
				_x_coord += _image_len

		_txt = [sg.TextElement(loc[0] + _image_len / 2, loc[1] + 1.1 * _image_len, str(n + 1), size=36, weight='bold', font='Arial') for loc, n in zip(_final_coords, _labels)]

		_fig.append(_plots)
		_fig.append(_txt)
		_fig.save(fname)