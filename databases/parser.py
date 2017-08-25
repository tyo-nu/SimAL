__author__ = 'Dante'

import re
import os
import pybel
import json

DBPATH = os.curdir

def severBRENDA(Brenda_dump):
	"""This function will check for the presence of individual enzyme files that will be scraped for data in the
	current directory.
    Arguments:

    Brenda_dump -- BRENDA data dump file name
    location -- current directory, which should contain the BRENDA data dump"""
	#Opens the full BRENDA dump file and reads it into a list, raw_data.
	input_file = open(Brenda_dump, 'r')
	raw_data = input_file.readlines()
	input_file.close()
	#Iteration through the dump list.  Each enzyme listing is separated by /// and so this will continue to separate
	# the enzymes into unique file s until there are no more ///s left.
	#As the unique files are created, the first /// in the list is deleted to ensure the loop ends.
	while '///\n' in raw_data:
		limit_index = raw_data.index('///\n')
		enzyme_name = raw_data[0].split('\t')[1]
		ec_number = enzyme_name.split(' ')[0].rstrip('\n')
		enzyme_file = open(os.getcwd() + '/' + ec_number + '.txt', 'w')
		enzyme_file.writelines(raw_data[:limit_index + 1])
		enzyme_file.close()
		del raw_data[:limit_index + 1]

def check_map(operator, mapping_file):
	"""This function will review mapping results (if they exist and are valid) and restrict the search to
	the EC
	numbers assigned in the mapping."""
	mapping_results = open(mapping_file, 'r')
	results_list = mapping_results.readlines()
	mapping_results.close()
	mapping = {}
	for entry in results_list:
		k = entry.split(':')[0]
		v = entry.split(':')[1].split(';')
		mapping[k] = v
	ecno_list = []
	x = re.compile(operator)
	for y in mapping.keys():
		match_test = x.match(y)
		if match_test and '' not in mapping[y]:
			for ecno in mapping[y]:
				ecno_list.append(ecno.strip('\n') + '.txt')
		if match_test and '' in mapping[y]:
				f = re.search('[a-z][0-9]*', y)
				z = y.strip(f.group(0))
				ecno_list.append(z.strip('\n') + '.txt')
	return ecno_list

def getECInfo(partialEc=None, location=DBPATH):
	'''This function will match all files that have an EC name and extract the information
	therein for use in subsequent functions.'''
	fileList = os.listdir(location)

	if partialEc:
		rgx = [re.compile(x) for x in partialEc]
		ecFilesA = []
		for x in rgx:
			for y in fileList:
				if re.match(x, y):
					ecFilesA.append(y)
		ecFiles = list(set(ecFilesA))
	else:
		rgx = re.compile('[0-9]\.[0-9]+\.[0-9]+\.[0-9]+\.txt')
		ecFiles = [filename for filename in fileList if re.match(rgx, filename)]

	ecInformation = {}

	for ecno in ecFiles:
		ecF = open(os.path.join(location, ecno), 'r')
		ecI = ecF.readlines()
		ecF.close()
		ecInformation[ecno.rstrip('.txt')] = ecI

	return ecInformation


def organism_pull(goodECInfo):

	ecOrgs = {x.split('\t')[1].split(' ')[0].strip('#'): ' '.join(x.split('\t')[1].split(' ')[1:]).split(' <')[0].strip(' ').split('  ')[0] for x in goodECInfo if x[0:3] == 'PR\t'}

	return ecOrgs


def cof_infos(data):

	uniqueCofIndices = sorted(list(set([data.index(l) for l in data if l[0:3] == 'CF\t'])))

	cofInfos = []

	for n in uniqueCofIndices:
		c = []
		i = 0
		for l in data[n:]:
			if i < 1 and l[0:3] == 'CF\t' or l == '\n':
				i += 1
				if re.search(r'\d$', l) is not None:
					if re.search(r'[A-Za-z \-]\d+$', l) is not None:
						c.append(l.strip('\n') + ' ')
					else:
						c.append(l.strip('\n') + ',')
				else:
					c.append(l.strip('\n') + ' ')
			if i <= 2 and 'CF\t' not in l and l != '\n':
				if re.search(r'\d$', l) is not None:
					if re.search(r'[A-Za-z \-]\d+$', l) is not None:
						c.append(l.strip('\n') + ' ')
					else:
						c.append(l.strip('\n') + ',')
				else:
					c.append(l.strip('\n') + ' ')
			if i >= 1 and l[0:3] == 'CF\t' or l == '\n':
				i += 1
		cofInfos.append(''.join([x.strip('\n\t') for x in c]))

	cofInfosFinal = []

	for d in cofInfos:
		if d[0:3] == 'CF\t':
			elements = d.split('#')
			olist = elements[1].split(',')
			cof_parts = []
			for piece in d.split(' ')[1:]:
				if '(' in piece or '<' in piece:
					break
				else:
					cof_parts.append(piece)
			cof = ' '.join(cof_parts)
			cofInfosFinal.append((olist, cof))

	return cofInfosFinal


def substrate_pull(goodECInfo):

	substratesTaken = {}
	productsMade = {}
	#This block identifies where specific reaction entries begin and stores them in uniqueReactionIndices.
	for ecNo in goodECInfo.keys():
		left = []
		right = []
		ecInfo = goodECInfo[ecNo]
		rxnInfoIndices = []
		for l in ecInfo:
			if l[0:4] == 'NSP\t':
				rxnInfoIndices.append(ecInfo.index(l))
			elif l[0:3] == 'SP\t':
				rxnInfoIndices.append(ecInfo.index(l))

		uniqueRxnInfoIndices = sorted(list(set(rxnInfoIndices)))
		#Joins the various lines of each entry together.
		rxnInfos = []
		for n in uniqueRxnInfoIndices:
			c = []
			i = 0
			for l in ecInfo[n:]:
				#Finds first line marked with leading NSP or SP indicator.
				if i < 1 and l[0:3] == 'SP\t' or l[0:4] == 'NSP\t' or l == '\n':
					i += 1
					if re.search(r'\d$', l) is not None:
						if re.search(r'[A-Za-z \-]\d+$', l) is not None:
							c.append(l.strip('\n') + ' ')
						else:
							c.append(l.strip('\n') + ',')
					else:
						c.append(l.strip('\n') + ' ')
				#Catches all subsequent lines in the entry.
				if i <= 2 and 'NSP\t' not in l and 'SP\t' not in l and l != '\n':
					if re.search(r'\d$', l) is not None:
						if re.search(r'[A-Za-z \-]\d+$', l) is not None:
							c.append(l.strip('\n') + ' ')
						else:
							c.append(l.strip('\n') + ',')
					else:
						c.append(l.strip('\n') + ' ')
				#Advances to the stop condition for all loops when it hits second line with leading NSP or SP.
				if i >= 1 and l[0:3] == 'SP\t' or l[0:4] == 'NSP\t' or l == '\n':
					i += 1
			#Makes the several lines of each entry into one line.
			rxnInfos.append(''.join([x.strip('\n\t') for x in c]))

		#Split the reaction entries and isolate the actual reaction expressions.
		j = 0
		for rxn in rxnInfos:
			j += 1
			a = rxn.lstrip('NSP\t')

			rxtElements = a.split(' = ')[0]
			prodElements = a.split(' = ')[1]

			#Separate into individual substrate/compound names.
			rxtants = rxtElements.split('# ')[1].lstrip('\t').split(' + ')
			prodSep = re.compile(' [|{(<]')
			products = re.split(prodSep, prodElements, 1)[0].lstrip('\t').split(' + ')

			#Filter out garbage strings and strip off transport designations.
			badEntries = ['More', 'more', '?']

			for s in rxtants:
				if re.search(r'/in$', s) is not None:
					left.append(s.rstrip('/in'))
				elif re.search(r'/out$', s) is not None:
					left.append(s.rstrip('/out'))
				elif s in badEntries:
					pass
				else:
					left.append(s)
			for s in products:
				if re.search(r'/in$', s) is not None:
					right.append(s.rstrip('/in'))
				elif re.search(r'/out$', s) is not None:
					right.append(s.rstrip('/out'))
				elif s in badEntries:
					pass
				else:
					right.append(s)

		substratesTaken[ecNo] = list(set(left))
		productsMade[ecNo] = list(set(right))

	return substratesTaken, productsMade

def screen_rxn(goodECInfo, strip_stoich=True):
	'''This function compares all of the reactions to the valid organism/cofactor combination to extract which reaction
	goes with the current query.  ***NB: works fine on Windows, but fails epically on Unix, even if you account for
	carriage returns.  Spacing gets all higgedly-piggedly.  FIX THIS EVENTUALLY.'''

	for ecNo in goodECInfo.keys():
		ecInfo = goodECInfo[ecNo]
		rxnInfoIndices = []
		for l in ecInfo:
			if l[0:4] == 'NSP\t':
				rxnInfoIndices.append(ecInfo.index(l))
			elif l[0:3] == 'SP\t':
				rxnInfoIndices.append(ecInfo.index(l))

		uniqueRxnInfoIndices = sorted(list(set(rxnInfoIndices)))

		rxnInfos = []
		for n in uniqueRxnInfoIndices:
			c = []
			i = 0
			for l in ecInfo[n:]:
				if i < 1 and l[0:3] == 'SP\t' or l[0:4] == 'NSP\t' or l == '\n':
					i += 1
					if re.search(r'\d$', l) is not None:
						if re.search(r'[A-Za-z \-]\d+$', l) is not None:
							c.append(l.strip('\n') + ' ')
						else:
							c.append(l.strip('\n') + ',')
					else:
						c.append(l.strip('\n') + ' ')
				if i <= 2 and 'NSP\t' not in l and 'SP\t' not in l and l != '\n':
					if re.search(r'\d$', l) is not None:
						if re.search(r'[A-Za-z \-]\d+$', l) is not None:
							c.append(l.strip('\n') + ' ')
						else:
							c.append(l.strip('\n') + ',')
					else:
						c.append(l.strip('\n') + ' ')
				if i >= 1 and l[0:3] == 'SP\t' or l[0:4] == 'NSP\t' or l == '\n':
					i += 1
			rxnInfos.append(''.join([x.strip('\n\t') for x in c]))

		rxnDict = {}
		j = 0
		for rxn in rxnInfos:
			j += 1

			# revflag = re.compile(" \{r\} ")
			# if re.search(revflag, rxn):
			# 	reverse = True
			# else:
			# 	reverse = False

			a = rxn.lstrip('NSP\t')
			oList = a.split(' ')[0].strip('#').split(',')
			rxtElements = a.split(' = ')[0]
			prodElements = a.split(' = ')[1]

			rxtants = rxtElements.split('# ')[1].lstrip('\t').split(' + ')
			prodSep = re.compile(' [|{(<]')
			products = re.split(prodSep, prodElements, 1)[0].lstrip('\t').split(' + ')

			badEntries = ['More', 'more', '?']
			left = []
			right = []
			leftside = []
			rightside = []
			stoich = []


			for s in rxtants:
				if re.search(r'/in$', s) is not None:
					left.append(s.rstrip('/in'))
				elif re.search(r'/out$', s) is not None:
					left.append(s.rstrip('/out'))
				elif s in badEntries:
					pass
				else:
					left.append(s)
			for s in products:
				if re.search(r'/in$', s) is not None:
					right.append(s.rstrip('/in'))
				elif re.search(r'/out$', s) is not None:
					right.append(s.rstrip('/out'))
				elif s in badEntries:
					pass
				else:
					right.append(s)

			for t in left:
				if re.search(r'(^[0-9]* )', t) is not None:
					leftside.append(re.split(r'(^[0-9]* )', t)[2])
					stoich.append('-' + re.split(r'(^[0-9]* )', t)[1].strip(' '))
				else:
					leftside.append(t)
					stoich.append('-1')
			for t in right:
				if re.search(r'^([0-9]* )', t) is not None:
					rightside.append(re.split(r'(^[0-9]* )', t)[2])
					stoich.append(re.split(r'(^[0-9]* )', t)[1].strip(' '))
				else:
					rightside.append(t)
					stoich.append('1')

			rxnDict[j] = (oList, leftside, rightside, stoich)

	return rxnDict

def TC_analysis(EC_dict, target_substrate, filenamechoice, smilesFile, inchiFile):
	"""This function carries out similarity indexing on the substrates of enzymes of interest.
    Arguments:

    EC_dict -- dictionary of EC numbers with the relevant substrates
    target_substrate -- SMILES string of the substrate of interest"""
	#Reading in the SMILES dump.
	smilesData = open(smilesFile)
	encodedSmilesDict = json.load(smilesData)
	smilesData.close()
	smilesDict = {k.encode('utf-8'): v.encode('utf-8') for k, v in encodedSmilesDict.iteritems()}

	inchiData = open(inchiFile)
	encodedInchiDict = json.load(inchiData)
	inchiData.close()
	inchiDict = {k.encode('utf-8'): v.encode('utf-8') for k, v in encodedInchiDict.iteritems()}

	net_substrate = pybel.readstring('smi', target_substrate)
	target_fp = net_substrate.calcfp('FP4')
	#Setting master_list to hold all the tuples of (TC, substrate, EC number)
	master_list = []
	for k, v in EC_dict.iteritems():
		#TC_vals = []
		substrates_sub = v

		for entry in substrates_sub:
			if entry in smilesDict:
				cString, z = smilesDict[str(entry)], 0
			elif entry in inchiDict:
				cString, z = inchiDict[str(entry)], 1
			else:
				continue
			#print str(entry)
			#Conversion to SMILES and TC calculation.
			if z == 0:
				try:
					instring = pybel.readstring('smi', cString)
					instring_fp = instring.calcfp('FP4')
					TC = round((instring_fp | target_fp), 3)
				except IOError:
					TC = float('-1')
			if z == 1:
				try:
					instring = pybel.readstring('inchi', cString)
					instring_fp = instring.calcfp('FP4')
					TC = round((instring_fp | target_fp), 3)
				except IOError:
					TC = float('-1')
			#Tuple creation in the order (TC, substrate, EC number).  A list of tuples can be sorted by whichever
			# entry, the default being the first.
			#We leave the default setting so that it sorts by descending TC and prints the tuples in that order to a
			# text file for our viewing pleasure.
			set_info = (TC, entry, k)
			master_list.append(set_info)
	final_list = sorted(master_list, reverse=True)
	stats_file = open(filenamechoice + '.txt', 'w')
	stats_file.write('BRENDA ENZYME SEARCH RESULTS\nRESULTS GIVEN AS:\nTC\tNATURAL SUBSTRATE\tEC NUMBER\n')
	for info in final_list:
		si, substr, ecno = info
		stats_file.write(str(si) + '\t' + substr + '\t' + ecno + '\n')
	stats_file.close()

def SimZyme(brenda_dumpfile, substrate, smilesDump, inchiDump, outName, ecNumbers, ecFlag, severFlag):

	dumpLocation = os.getcwd() + '/' + brenda_dumpfile

	if severFlag:
		severBRENDA(dumpLocation)

	if ecFlag:
		ecInformation = getECInfo(ecNumbers)
		substrateEcDict, productEcDict = substrate_pull(ecInformation)
		TC_analysis(substrateEcDict, substrate, outName, smilesDump, inchiDump)
	else:
		mapResults = check_map(ecNumbers, ecFlag)
		ecInformation = getECInfo(mapResults)
		substrateEcDict, productEcDict = substrate_pull(ecInformation)
		TC_analysis(substrateEcDict, substrate, outName, smilesDump, inchiDump)


if __name__ == "__main__":

	i = getECInfo(partialEc=['1.4.1.20'])
	print screen_rxn(i)




