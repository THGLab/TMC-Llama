import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import time
from tqdm import tqdm
import os

class InferenceObject:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to('cuda')  # We'll need to manage device assignment carefully

    def create_prompt(
        self, 
        user_txt="Output a SMILES string for a drug like molecule:", 
        system_txt="You love and excel at generating SMILES strings of drug-like molecules"):
        messages = [{"role": "system", "content": system_txt}, {"role": "user", "content": user_txt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate_single_batch(self, prompts, num_return_sequences, max_new_tokens, gpu_id, generation_params={}):
        # Set the GPU for this process
        torch.cuda.set_device(gpu_id)
        self.model.to(f'cuda:{gpu_id}')
        
        tokenized_prompts = [self.tokenizer(p, padding=True, return_tensors="pt") for p in prompts]
        completions = []
        
        for batch in tokenized_prompts:
            batch = {k: v.to(f'cuda:{gpu_id}') for k, v in batch.items()}
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_params
            )
            cut_outputs = outputs[:, len(batch["input_ids"][0]):]
            generated_text = self.tokenizer.batch_decode(cut_outputs, skip_special_tokens=True)
            completions.extend(generated_text)
            
        return completions

def worker_process(rank, model_path, prompts, num_return_sequences, max_new_tokens, generation_params, result_queue):
    try:
        inference_obj = InferenceObject(model_path, model_path)
        results = inference_obj.generate_single_batch(
            prompts, 
            num_return_sequences, 
            max_new_tokens,
            rank,
            generation_params
        )
        result_queue.put((rank, results))
    except Exception as e:
        result_queue.put((rank, f"Error in process {rank}: {str(e)}"))

if __name__ == "__main__":
    # Initialize multiprocessing with spawn method for CUDA support
    mp.set_start_method('spawn')
    
    sl_path = '/global/scratch/users/jmcavanagh/smiley/models/SmileyLlama-3.1-8B-Instruct'
    
    # Create prompts
    temp_obj = InferenceObject(sl_path, sl_path)
    prompt = temp_obj.create_prompt()
    prompts = [prompt, prompt, prompt]
    
    # Set up multiprocessing
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")
    
    # Split prompts among available GPUs
    prompts_per_gpu = [prompts[i::num_gpus] for i in range(num_gpus)]
    
    # Create a queue for results
    result_queue = mp.Queue()
    
    # Start processes
    processes = []
    for rank in range(num_gpus):
        if len(prompts_per_gpu[rank]) > 0:  # Only start process if there are prompts to process
            p = mp.Process(
                target=worker_process,
                args=(rank, sl_path, prompts_per_gpu[rank], 5, 256, {"top_k": 3}, result_queue)
            )
            p.start()
            processes.append(p)
    
    # Collect results
    all_results = []
    for _ in range(len(processes)):
        rank, result = result_queue.get()
        if isinstance(result, str) and result.startswith("Error"):
            print(result)
        else:
            all_results.extend(result)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Print results
    for s in all_results:
        print(s.rstrip())
    print(len(all_results))
