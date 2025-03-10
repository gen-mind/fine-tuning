import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

def load_model_and_tokenizer(model_id, cache_dir="cache"):
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
        cache_dir=cache_dir,
    )
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        bnb_4bit_quant_compute_dtype=model_kwargs['torch_dtype'],
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def stream(model, user_prompt, model_type, tokenizer, checkpoint=""):
    if model_type == "base":
        eval_model = model
    elif model_type == "fine-tuned":
        # Load adapter from local folder using its absolute path.
        eval_model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
        eval_model = eval_model.to("cuda")
    else:
        print("You must set the model_type to 'base' or 'fine-tuned'")
        exit()

    eval_model.config.use_cache = True
    messages = [{"role": "user", "content": user_prompt.strip()}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([inputs], return_tensors="pt", add_special_tokens=False).to("cuda")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    streamer = TextStreamer(tokenizer)
    print(f"Model device: {next(eval_model.parameters()).device}")
    print(f"Input IDs device: {inputs['input_ids'].device}")

    _ = eval_model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.cuda.empty_cache()
    gc.collect()

def evaluation(model, model_type, tokenizer, checkpoint=""):
    questions = [
        "In the context of hot air balloon, What can be inferred about the effectiveness of air as a conductor of heat based on the text?",
        "In what ways does ICAO influence the growth of international air transport?"
    ]
    for question in questions:
        print("\n" + "=" * 50)
        print("User Question:", question)
        stream(model, question, model_type, tokenizer, checkpoint)
        print("\n" + "=" * 50 + "\n")

def main():
    print("\n" + "*" * 150)
    print("Evaluating the Base Model:")
    print("\n" + "*" * 150)
    model_id = "Qwen/Qwen1.5-7B-Chat"
    model, tokenizer = load_model_and_tokenizer(model_id)
    evaluation(model, "base", tokenizer)

    # Convert relative path to an absolute path.
    # ft_checkpoint = os.path.abspath("Qwen1.5-7B-Chat-test-gian-local")
    print("\n" + "*" * 150)
    print("Evaluating the Fine-Tuned Model:")
    print("\n" + "*" * 150)
    # evaluation(model, "fine-tuned", tokenizer, checkpoint=ft_checkpoint)
    model_id = "Qwen1.5-7B-Chat-test-gian-merged-local"
    model, tokenizer = load_model_and_tokenizer(model_id)
    evaluation(model, "base", tokenizer)

if __name__ == "__main__":
    main()
