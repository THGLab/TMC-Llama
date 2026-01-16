'''
This runs the guacamol suite of tests on some SmileyLlama-like LLM.
In particular, it runs inference, creating files of 20,000 SMILES strings in some directory
Then, it analyzes each file using the guacamol analysis in guacamol_minimal
'''
import sl_inference
import guacamol_minimal
import os
from accelerate import Accelerator

def inference_and_analysis(llm_inference_obj, smilesfile, generation_params, chembl_file):
    prompts = [llm_inference_obj.create_prompt()]*80
    completions = []
    generation = llm_inference_obj.generate_smiles_strings(prompts, 256, 256, generation_params)
    if generation is not None:
        completions = generation
        print(f"Generated {len(completions)} completions")
        #Truncate completions to 20,000
        assert len(completions) >= 20000
        completions = completions[:20000]
        #Write to smilesfile
        with open(smilesfile, 'w+') as f:
            for c in completions:
                f.write(c.strip() + '\n')
        #Perform guacomol analysis
        '''
        print("Performing guacamol analysis on", smilesfile)
        guac_results = guacamol_minimal.run_benchmarks(chembl_file,smilesfile)
        total_score = guac_results['Validity']*guac_results['Uniqueness']*guac_results['Novelty']
        guac_results['Score'] = total_score
        print(guac_results)
        return guac_results
        '''
    return None

if __name__=="__main__":
    accelerator = Accelerator()
    chembl_file = '/global/scratch/users/jmcavanagh/smiley/data/rdkit_can_iso_smiles.txt'
    logfile = 'logfile.txt'
    '''
    projects = ['chembl_original_smiles-1b-llama-ft', 
    'rdkit_can_iso_smiles-1b-llama-ft', 
    'rdkit_can_iso_smiles_h-1b-llama-ft',
    'rdkit_can_kek_iso_smiles-1b-llama-ft',
    'rdkit_can_kek_iso_smiles_h-1b-llama-ft',
    'rdkit_noncan_smiles-1b-llama-ft',
    'rdkit_random_smiles-1b-llama-ft']
    temperatures = [0.1*n for n in range(5, 15)]
    '''
    projects = ['rdkit_can_iso_smiles-1b-llama-ft']
    temperatures = [0.9, 1.0]
    project_dirs = ['/global/scratch/users/jmcavanagh/smiley/typesmiles/1b_prep/' + p for p in projects]
    for p in project_dirs:
        #Make a directory for the guacamol analysis if it doesn't exist
        guac_dir = p + '/guac_analysis'
        if not os.path.exists(guac_dir) and accelerator.is_main_process:
            os.makedirs(guac_dir)
            print(p, file=open(logfile, 'a'))
        #Create the inference object and load the model
        sl_path = p + '/outputs/merged/'
        sl = sl_inference.InferenceObject(sl_path, '/global/scratch/users/jmcavanagh/smiley/models/SmileyLlama-3.1-8B-Instruct')
        for t in temperatures:
            smilesfile = p + '/guac_analysis/temp' + str(t) + '.txt'
            generation_params = {"temperature":t}
            guac_results = inference_and_analysis(sl, smilesfile, generation_params, chembl_file)
            # Wait for all processes to sync
            accelerator.wait_for_everyone()
            #Print the temperature and results
            if guac_results is not None:
                print(t, guac_results, file=open(logfile, 'a'))