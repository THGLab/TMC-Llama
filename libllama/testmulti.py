import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing as mp
from typing import List, Tuple
from tqdm import tqdm

def load_model(args):
    gpu_id, model_path = args
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        device_map={'': device}
    )

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    return tokenizer, model, stopping_ids

def process_batch(args):
    tokenizer, model, stopping_ids, gpu_id, prompt = args
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            num_return_sequences=3,
            eos_token_id=stopping_ids,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        generated_texts.append(generated_text.strip())
    return generated_texts

def main(model_path: str, prompts: List[str], batch_size: int = 32):
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    if num_gpus > 1:
        # Initialize multiprocessing
        mp.set_start_method('spawn', force=True)

        # Load models in parallel
        with mp.Pool(num_gpus) as pool:
            model_args = [(gpu_id, model_path) for gpu_id in range(num_gpus)]
            models = pool.map(load_model, model_args)

        # Create process pool for inference
        pool = mp.Pool(num_gpus)

        # Distribute prompts across GPUs
        all_args = []
        for i, prompt in enumerate(prompts):
            gpu_id = i % num_gpus
            tokenizer, model, stopping_ids = models[gpu_id]
            all_args.append((tokenizer, model, stopping_ids, gpu_id, prompt))

        # Process prompts in parallel
        results = []
        for batch_start in range(0, len(all_args), batch_size):
            batch_args = all_args[batch_start:batch_start + batch_size]
            batch_results = pool.map(process_batch, batch_args)
            results.extend(batch_results)

        pool.close()
        pool.join()

        return results

    else:
        # Single GPU/CPU processing
        tokenizer, model, stopping_ids = load_model((0, model_path))
        results = []

        for prompt in tqdm(prompts):
            result = process_batch((tokenizer, model, stopping_ids, 0, prompt))
            results.append(result)

        return results


if __name__ == "__main__":
    model_path = "/global/scratch/users/jmcavanagh/smiley/models/SmileyLlama-3.1-8B-Instruct"
    prompts = ["Once upon a time"] * 100  # Example with 100 identical prompts
    
    results = main(model_path, prompts)
    
    # Process results
    for i, generations in enumerate(results):
        print(f"\nPrompt {i} generations:")
        for j, text in enumerate(generations):
            print(f"Generation {j}: {text}")
