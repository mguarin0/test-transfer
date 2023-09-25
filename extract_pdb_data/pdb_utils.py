#!/usr/bin/env python3

"""
Created on Fri Oct 5 9:30:10 2018

@author: danielburkat
"""

import os
import re
import csv
import pandas as pd


def count_pdb_fp():
    fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb'
    dir_c = 0
    files_c = 0

    for path, dirs, files in os.walk(fp):
        dir_c += 1
        for name in files:
            files_c += 1

    print(dir_c)
    print(files_c)


def run_extract_source():

    fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb'

    # Create the file that will store the data from the visited pdb entry files
    extracted_source_fp = '/home/cc/workspace_db/extract_pdb_data/extracted_source_raw.csv'
    with open(extracted_source_fp, 'w') as extract_f:
        #writer = csv.writer(extract_f)
        #writer.writerow(['fp','source'])
        extract_f.write("fp, source\n")

    with open(extracted_source_fp, 'a') as extract_f:

        file_c = 0
        for path, dirs, files in os.walk(fp):
            for name in files:
                if name[-4:] == ".swp":
                    continue   
                pdb_entry_fp = os.path.join(path,name)
                extract_f.write(pdb_entry_fp[-14:]+", ")
                extract_f.write(find_source_v2(pdb_entry_fp)+"\n")
                file_c += 1
                if file_c % 1000 == 0: 
                    print(file_c)


def find_source_v2(pdb_id_fp):

    #This will be the file where the data is dumped
    extracted_source_fp = '/home/cc/workspace_db/extract_pdb_data/extracted_source_raw.csv'

    pat_source = re.compile(r'SOURCE')

    source_text = ""
    with open(pdb_id_fp, 'r') as f:

        line = f.readline()
        reached_source = False
        while line:
            match = re.match(pat_source, line)
            # If matched on source then extract the text on that line
            if match:
                reached_source = True
                source_text += line[10:].strip()
            
            if not match and reached_source:
                break

            line = f.readline()
    
    return source_text


def find_source(pdb_id_fp):
    #This will be the file where the data is dumped
    extracte_source_fp = '/home/cc/workspace_db/extract_pdb_data/extracted_source_temp.csv'
    
    # We are looking for the lines that start with the following pattern
    pat_head = re.compile(r'HEADER')
    pat_source = re.compile(r'SOURCE')

    source_data = []
    
    with open(pdb_id_fp, 'r') as f:

        #The header should be in the first line
        line = f.readline()
        match = re.match(pat_head, line)
        if match:
            pdb_id = line[62:66]
        else:
            print("no header for fp: "+pdb_id_db)
            return



        #Find the SOURCE lines
        line = f.readline()
        reached_source = False
        first_mol = True
        org_tax = ""
        gene_list = [""]
        while line:
            
            match = re.match(pat_source, line)
            # If you match on SOURCE then do the following            
            if match:
                reached_source = True
                
                #Do the source extract sequence
                #First look for the molecule ID
                mol_match = re.search(r'MOL_ID',line[10:])
                if mol_match:
                    #Could be more than one macromolecule per pdb entry    
                    if not first_mol:
                        for gene in gene_list:
                            source_data.append([pdb_id,mol_id, org_tax, gene])
                        org_tax = ""
                        gene_list = [""]

                    first_mol = False
                    # extract the number
                    m = re.search(r'(?P<mol_id>\d+);',line[16:])
                    if m:
                        mol_id = m.group('mol_id')

                # Get the organism_common filed
                if line[11:25] == "ORGANISM_TAXID":
                    m = re.search(r'(?P<org_t>\d+);',line[25:])
                    if m:
                        org_tax = m.group('org_t')
                
                # Get all the gene symbols in this line
                if line[11:15] == "GENE":
                    m = re.search(r'(?P<g>[\w\s.,]+);',line[15:])
                    if m:
                        genes = m.group('g')
                        genes = genes.replace(" ","")
                        gene_list = genes.split(',')

            if not match and reached_source:
                # Append the macromolecule to the pdb entry
                for gene in gene_list:
                    source_data.append([pdb_id, mol_id, org_tax, gene])
                #Break out of this loop
                break

            # Read in the line for the next iteration
            line = f.readline()

    # Write the source_data to a file
    with open(extracte_source_fp, 'a') as write_source_f:
         writer = csv.writer(write_source_f)
         for row in source_data:
            writer.writerow(row)




def run_extract_dbref():

    fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb'
    
    #Create the folder that will store the data from the visited pdb entry files
    extracted_dbref_fp = '/home/cc/workspace_db/extract_pdb_data/extracted_dbref.csv'
    with open(extracted_dbref_fp, 'w') as extract_f:
        writer = csv.writer(extract_f)
        writer.writerow(['id','chain_id','db','seq_db_acc','seq_id','fp'])

    file_c = 0
    for path, dirs, files in os.walk(fp):
        for name in files:
            find_dbref(os.path.join(path, name))
            file_c += 1
            if file_c % 1000 == 0: 
                print(file_c)



def find_dbref(pdb_id_fp):
    #This will be the file to dump all the data
    extracted_dbref_fp = '/home/cc/workspace_db/extract_pdb_data/extracted_dbref.csv'
    
    # The lines we want start with the following pattern
    pattern = re.compile(r'DBREF')

    # In this list store the lines with DBREF for this file
    dbref_data = []

    with open(pdb_id_fp, 'r') as f:

        start_dbref = False
        passed_dbref = False
        
        line = f.readline()

        while line and (not passed_dbref):

            match = re.match(pattern, line)
            if match:
                # This flag is used to break out of the while loop once we have passed all the 
                # lines with DBREF
                start_dbref = True
                
                # Extract the data
                idcode = line[7:11].strip()
                chainID = line[12].strip()
                database = line[26:32].strip()
                seqdbcode = line[33:41].strip()
                seqdbid = line[42:54].strip()

                dbref_data.append([idcode, chainID, database, seqdbcode, seqdbid, pdb_id_fp])

            # Go in here when you have passed all the lines with DBREF. Sets us to break out of
            # the while loop
            if (match is None) and start_dbref:
                passed_dbref = True

            # Read in the next line for the following iteration
            line = f.readline()

    #Now save all the dbref_data lines to the file
    with open(extracted_dbref_fp, 'a') as f:
        writer = csv.writer(f)
        for row in dbref_data:
            writer.writerow(row)



# Restrict the lines in extract_pdb_data to only those that contain HUMAN, MOUSE, and RAT genes
def process_dbref_lines():

    extracted_dbref_fp = '/home/cc/workspace_db/extract_pdb_data/extracted_dbref.csv'

    dbref = pd.read_csv(extracted_dbref_fp)
    seq_id = dbref['seq_id']
    
    sis = seq_id.str.split('_', expand=True)
    sis.columns = ['gene', 'species']

    dbref['gene'] = sis['gene']
    dbref['species'] = sis['species']
    dbref.drop('seq_id',axis=1, inplace=True)

    #hrm = ['HUMAN','RAT','MOUSE']

    #hrm_dbref = dbref.loc[sis.species.isin(hrm)]

    #save to a file called hrm_dbref.csv
    #fp = '/home/cc/workspace_db/extract_pdb_data/dbref_gene_species.csv'
    #dbref.to_csv(fp)


# Locate the DBREF lines that match to an entry_id
def get_dbref_at_id(id_code):
    
    dbref_gene_species_fp = '/home/cc/workspace_db/extract_pdb_data/dbref_gene_species.csv'

    dbref = pd.read_csv(dbref_gene_species_fp)
    match_id = dbref.loc[dbref['id'] == id_code]

    # save to a temporary file so you can check if it works
    temp_output_fp = '/home/cc/workspace_db/extract_pdb_data/temp_output.csv'
    match_id.to_csv(temp_output_fp)




if __name__ == "__main__":
    #get_dbref_at_id('2B42')
    #run_extract_source()
    #fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/ob/pdb5ob4.ent'
    #find_source(fp)
    #fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/nn/pdb4nnd.ent'
    #find_source(fp)
    #fp = '/home/cc/rawchem/pdb/ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/nn/pdb4nng.ent'
    #find_source(fp)
