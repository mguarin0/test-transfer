#!/usr/bin/env python3

"""
Created on Tuesday Oct 18 15:22:00 2018

@author: danielburkat
"""


import re
import csv
import pandas as pd


#This func will build a csv file that will match an pdbid and a mol_id to an AA sequence
def get_list_of_aa_seq(mol_list, csv_fp):
    extracted_data_folder = '/home/cc/workspace_db/extract_pdb_data/'
    
    #First build the list, then save it all    
    out_list = []
    for row in mol_list:
        gene = row[0]
        pdb_id = row[1]
        mol_id = row[2]
        fp = row[3]
        print(gene, mol_id, fp)
        aa_seq = get_aa_seq(mol_id, fp)
        out_list.append([gene, pdb_id, mol_id, fp, aa_seq])

    #Save what was extracted to a file
    with open(extracted_data_folder+csv_fp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['gene','pdbid', 'molid', 'fp', 'aa_seq'])
        for row in out_list:
            writer.writerow(row)




def get_aa_seq(mol_id, fp):

    full_fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/' + fp
    
    compnd_text = find_compnd(fp)
    chain_list = find_molid_chain(mol_id, compnd_text)
    if len(chain_list) == 0:
        return None

    #If there are more than one chain then for now just choose the first one
    chain = chain_list[0]

    # Go to the SEQRES and extract the 
    aa_seq = find_chain_seqres(chain, full_fp)
    aa_list = aa_seq.split(" ")
    
    #This dictionary conversion can be somewhere else. For now do it in this fucntion
    one_letter_aa = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "ASX": "B",
            "CYS": "C",
            "GLU": "E",
            "GLN": "Q",
            "GLX": "Z",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V"
            }
    
    aa_one_letter_seq = ""
    try:
        for aa in aa_list:
            aa_one_letter_seq += one_letter_aa[aa]
    except KeyError:
        aa_one_letter_seq = 'keyerror'

    return aa_one_letter_seq




    
def find_chain_seqres(chain, f_fp):
    

    pat_seqres = re.compile(r'SEQRES')

    aa_seq = ""
    with open(f_fp, 'r') as f:

        line = f.readline()
        reached_seqres = False
        while line:
            match = re.match(pat_seqres, line)
            # if matched then extract the text on that line
            if match:
                reached_seqres = True
                if line[11] == chain:
                    aa_seq = aa_seq + line[19:].strip() + " "
            if not match and reached_seqres:
                break

            line = f.readline()

    return aa_seq.strip()



def find_molid_chain(mol_id, compnd_text):
    
    # Helper function to get the locations to mark the end of one mol_id
    # Made it as a recursive function
    def get_end_loc(loc, compnd_text_end):
        #Base case: loc < 1, the end loc is simply source_end
        if len(loc)  < 2:
            return [compnd_text_end]
        else:
            end_loc_list = get_end_loc(loc[1:], compnd_text_end)
            cur_end= loc[1] - 8
        return [cur_end] + end_loc_list



    pat_mol = re.compile(r'MOL_ID:')
    loc = [match.end() for match in re.finditer(pat_mol, compnd_text)]

    if len(loc) < 1:
        return None
    end_loc = get_end_loc(loc, len(compnd_text))
    
    #with loc and end_loc I now have the limits to look at each mol_id separately
    chain = []
    for index, val in enumerate(loc):
        mol_entry = compnd_text[val:end_loc[index]].strip().split(';')
        if mol_entry[0] == mol_id:
            for entry in mol_entry[1:]:
                if (len(entry) > 6) and (entry[0:5] == "CHAIN"):
                    chain = entry[6:].strip()


    return chain
    



# There is the chain infomration in the COMPND rows that I need
# One mol_id that was found could have multiple chains. 
def find_compnd(fp):

    full_fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/' + fp

    pat_compnd = re.compile(r'COMPND')

    comp_text = ""

    with open(full_fp, 'r') as f:

        line = f.readline()
        reached_compnd = False
        while line:
            match = re.match(pat_compnd, line)
            #If match, then add that lines text (omit the "COMPND  #")
            if match:
                reached_compnd = True
                comp_text += line[10:].strip()

            if not match and reached_compnd:
                break

            line = f.readline()

    return comp_text



if __name__ == "__main__":
    #fp = 'bw/pdb5bwo.ent'
    #mol_id = '2'
    #compnd_text = find_compnd(fp)
    #print(compnd_text)

    #chain = find_molid_chain(mol_id, compnd_text)
    #print(chain)

    #full_fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/' + fp
    #aa_seq = find_chain_seqres(chain, full_fp)
    #print(aa_seq)
    
    #aa = get_aa_seq('', mol_id, fp)
    #print(aa)


    #Load a mol_list that is requested
    mol_list = []
    fp = 'gene_to_get_seq.csv'
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        mol_list = list(reader)

    target_fp = 'genes_with_aa_seq.csv'
    get_list_of_aa_seq(mol_list, target_fp)











