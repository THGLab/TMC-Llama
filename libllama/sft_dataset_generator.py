from rdkit import Chem
import json, random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import property_tools
from functools import partial


def create_prompt(sm_str):
    template = {
    "instruction": "You love and excel at generating SMILES strings of drug-like molecules",
    "input":None,
    "output": None
    }
    mol = Chem.MolFromSmiles(sm_str)
    if not mol:
         return {}
    else:
        kwargs = {'hbd_range': (0.5 > random.random())}
        kwargs['hba_range'] =  (0.5 > random.random())
        kwargs['mw_range'] =  (0.5 > random.random())
        kwargs['logp_range'] =  (0.5 > random.random())
        kwargs['rotb_range'] =  (0.5 > random.random())
        kwargs['fracsp3_range'] =  (0.5 > random.random())
        kwargs['tpsa_range'] =  (0.5 > random.random())
        kwargs['macrocycle'] =  (0.5 > random.random())
        kwargs['formula'] =  (0.5 > random.random())
        kwargs['undesirable_smarts'] = (0.5 > random.random())
        kwargs['cov_warhead'] = (0.5 > random.random())
        kwargs['substruct'] = (0.5 > random.random())
    #print(kwargs)
    input = property_tools.generate_inp_prompt(mol, **kwargs)
    return {**template, "output":sm_str, "input":input}

def parallelize_processing(sm_strs):
    sm_strs = list(set(sm_strs))
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(create_prompt, sm_strs), total=len(sm_strs)))
    return results

def convert_smiles_list(sm_strs, doRandom, canonical, kekuleSmiles, allHsExplicit):
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    print("Number of cores: ", num_cores)
    
    # Create argument tuples for each SMILES string
    args = [(sm_str, doRandom, canonical, kekuleSmiles, allHsExplicit) 
            for sm_str in sm_strs]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(convert_smiles_string, args), 
                          total=len(sm_strs)))
    
    # Remove None values (failed conversions)
    results = [r for r in results if r is not None]
    return results

def convert_smiles_string(args):
    # Unpack all arguments from the tuple
    sm_str, doRandom, canonical, kekuleSmiles, allHsExplicit = args
    try:
        mol = Chem.MolFromSmiles(sm_str)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, doRandom=doRandom,
                               canonical=canonical,
                               kekuleSmiles=kekuleSmiles,
                               allHsExplicit=allHsExplicit)
    except:
        return None


def combinatorial_dataset_generator(sm_strs, epochs, output_dir):
    #Not implemented yet, but a lot more elegant than the current method.
    run_parameters = {
        "random": [True, False, False, False],
        "random_kekule": [True, False, True, False],
        "random_explicit": [True, False, False, True],
        "random_kekule_explicit": [True, False, True, True],
        "canonical": [False, True, False, False],
        "canonical_kekule": [False, True, True, False],
        "canonical_explicit": [False, True, False, True],
        "canonical_kekule_explicit": [False, True, True, True],
        "noncanonical": [False, False, False, False],
        "noncanonical_kekule": [False, False, True, False],
        "noncanonical_explicit": [False, False, False, True],
        "noncanonical_kekule_explicit": [False, False, True, True]
    }
    for param in run_parameters:
        for epoch in range(epochs):
            sm_strs = convert_smiles_list(sm_strs, **run_parameters[param])
            jsondata = parallelize_processing(sm_strs)
            outputfile = output_dir+"/"+param+"_"+str(epoch)+"epochs.jsonl"
            with open(outputfile, 'w+') as out:
                for item in jsondata:
                    out.write(json.dumps(item) + '\n')

def main():
    initial_smiles = []
    df = pd.read_csv('/global/scratch/users/jmcavanagh/smiley/data/oliver_chembl33_data/chembl_33_filtered_processed.csv')
    """
    for col_name in df.columns[1:]:
        print(col_name)
        sm_strs = list(df[col_name])
        jsondata = parallelize_processing(sm_strs)
        counter = 0
        for p in jsondata:
            if 'lacks covalent warheads' in p['input']:
                counter +=1
        print(counter)
        outputfile = "/global/scratch/users/jmcavanagh/smiley/data/revised_sft_datasets/"+col_name+".jsonl"
        with open(outputfile, 'w+') as out:
            for item in jsondata:
                out.write(json.dumps(item) + '\n')
        jsondata += parallelize_processing(sm_strs)
        outputfile = "/global/scratch/users/jmcavanagh/smiley/data/revised_sft_datasets/"+col_name+"_2epochs.jsonl"
        with open(outputfile, 'w+') as out:
            for item in jsondata:
                out.write(json.dumps(item) + '\n')
    """
    sm_strs = list(df["rdkit_can_iso_smiles"])
    sm_strs = convert_smiles_list(sm_strs, doRandom=True, canonical=False, kekuleSmiles=False, allHsExplicit=False)
    jsondata = parallelize_processing(sm_strs)
    outputfile = "/global/scratch/users/jmcavanagh/smiley/data/revised_sft_datasets/random_smiles.jsonl"
    with open(outputfile, 'w+') as out:
        for item in jsondata:
            out.write(json.dumps(item) + '\n')
    sm_strs = convert_smiles_list(sm_strs, doRandom=True, canonical=False, kekuleSmiles=False, allHsExplicit=False)
    jsondata += parallelize_processing(sm_strs)
    outputfile = "/global/scratch/users/jmcavanagh/smiley/data/revised_sft_datasets/random_smiles_2epochs.jsonl"
    with open(outputfile, 'w+') as out:
        for item in jsondata:
            out.write(json.dumps(item) + '\n')
    
    sm_strs = list(df["rdkit_can_iso_smiles"])
    sm_strs = convert_smiles_list(sm_strs, doRandom=True, canonical=False, kekuleSmiles=True, allHsExplicit=False)
    jsondata = parallelize_processing(sm_strs)
    counter = 0
    for p in jsondata:
        if 'lacks covalent warheads' in p['input']:
            counter +=1
    print(counter)

    outputfile = "/global/scratch/users/jmcavanagh/smiley/data/revised_sft_datasets/random_kekule_smiles.jsonl"
    with open(outputfile, 'w+') as out:
        for item in jsondata:
            out.write(json.dumps(item) + '\n')
    
    
    sm_strs = convert_smiles_list(sm_strs, doRandom=True, canonical=False, kekuleSmiles=True, allHsExplicit=False)
    jsondata += parallelize_processing(sm_strs)
    outputfile = "/global/scratch/users/jmcavanagh/smiley/data/revised_sft_datasets/random_kekule_smiles_2epochs.jsonl"
    with open(outputfile, 'w+') as out:
        for item in jsondata:
            out.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
