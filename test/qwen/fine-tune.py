import torch
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# region configuration
model_id = "Qwen/Qwen1.5-7B-Chat"
fine_tune_tag = "faa-balloon-flying-handbook"
cache_dir = "cache"

# if true Main will execute login to hf and il will then upload the model to HF
upload_to_hf = False

# Set up LoRA configuration and wrap the model
peft_config = LoraConfig(
    r=16,
    modules_to_save=["lm_head", "embed_tokens"],
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

system_message = """Answer the given Balloon Flying Handbook question by providing a clear, detailed explanation that references guidance from the Balloon Flying Handbook, operational procedures, and relevant flight concepts.

Provide a detailed breakdown of your answer, beginning with an explanation of the question and its context within the Balloon Flying Handbook, followed by step-by-step reasoning based on the information provided in the Handbook and applicable flight operation procedures. Use logical steps that build upon one another to arrive at a comprehensive solution.

# Steps

1. **Understand the Question**: Restate the given question and clearly identify the main query along with any relevant details about balloon operations, safety procedures, or flight scenarios as discussed in the Balloon Flying Handbook.
2. **Handbook Context**: Explain the relevant procedures and guidelines as outlined in the Balloon Flying Handbook. Reference specific sections of the Handbook, such as pre-flight checks, flight planning, emergency procedures, and operational parameters central to the question.
3. **Detailed Explanation**: Provide a step-by-step breakdown of your answer. Describe how you arrived at each conclusion by citing pertinent sections of the Handbook and relevant operational standards.
4. **Double Check**: Verify that your explanation is consistent with the guidelines in the Balloon Flying Handbook and accurate according to current practices. Mention any alternative methods or considerations if applicable.
5. **Final Answer**: Summarize your answer clearly and concisely, ensuring that it is accurate and fully addresses the question.

# Notes

- Clearly define any terms or procedures specific to balloon flight operations as described in the Handbook.
- Include relevant procedural steps, operational parameters, or safety guidelines where applicable to support your answer.
- Assume a familiarity with basic flight operation concepts while avoiding overly technical jargon unless it is commonly used in the ballooning community.
"""


# endregion
def load_model_and_tokenizer(model_id, cache_dir):
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        # bf16=True,
        # tf32=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
        cache_dir=cache_dir,
    )
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }


def main():
    if upload_to_hf:
        # login to HF
        login(
            token=os.getenv('HF_TOKEN'),  # ADD YOUR TOKEN HERE
            add_to_git_credential=True
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)

    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)

    # load and preprocess the dataset
    dataset_id = "gsantopaolo/faa-balloon-flying-handbook"
    train_dataset = load_dataset(dataset_id, split="train")
    validation_dataset = load_dataset(dataset_id, split="validation")

    # apply template
    train_dataset = train_dataset.map(create_conversation, remove_columns=train_dataset.features, batched=False)
    validation_dataset = validation_dataset.map(create_conversation, remove_columns=validation_dataset.features,
                                                batched=False)
    train_dataset = train_dataset.take(1000)

    def tokenize(sample):
        # For batched mapping, sample["messages"] is a list of conversations.
        conversation_strs = []
        for conversation in sample["messages"]:
            # conversation is a list of message dictionaries.
            conv_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
            conversation_strs.append(conv_str)
        # Tokenize the list of conversation strings.
        tokenized = tokenizer(conversation_strs, truncation=True, max_length=1024)
        # Add the raw text as a new field that SFTTrainer expects.
        tokenized["text"] = conversation_strs
        return tokenized

    tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(
        [col for col in tokenized_train_dataset.column_names if col not in ["input_ids", "attention_mask", "text"]]
    )

    tokenized_validation_dataset = validation_dataset.map(tokenize, batched=True)

    model_name = model_id.split("/")[-1]
    dataset_name = dataset_id.split("/")[-1]

    context_length = 1024

    save_dir = f"./results/{model_name}_{dataset_name}_1_epochs_{context_length}_length-{fine_tune_tag}"
    print("Save directory:", save_dir)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        args=TrainingArguments(
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-6,
            # lr_scheduler_type="cosine",
            lr_scheduler_type="constant",
            warmup_ratio=0.1,
            save_steps=50,
            bf16=True,

            # eval
            per_device_eval_batch_size=2,
            # evaluation_strategy="steps", "epoc" # set to "no" to skip the eval_dataset
            evaluation_strategy="steps",
            do_eval=True,
            eval_steps=50,

            # Logging arguments
            logging_strategy="steps",
            logging_steps=5,
            report_to=["tensorboard"],
            save_strategy="epoch",
            seed=42,
            output_dir=save_dir,
            log_level="debug",
        ),
    )

    model.config.use_cache = False

    trainer.train()

    if upload_to_hf:
        # upload to HF
        adapter_model = f"gsantopaolo/{model_name}-{fine_tune_tag}-adapters"
        new_model = f"gsantopaolo/{model_name}-{fine_tune_tag}"

        # This saves the model with its adapter weights still separate (the PEFT model).
        # It preserves the adapter configuration so that it can be loaded as an adapter later.
        model.save_pretrained(f"{model_name}-{fine_tune_tag}", push_to_hub=True, use_auth_token=True)
        model.push_to_hub(adapter_model, use_auth_token=True, max_shard_size="10GB", use_safetensors=True)

        from huggingface_hub import HfApi, create_repo, create_branch

        create_repo(new_model, private=True)
        create_branch(new_model, repo_type="model", branch="gguf")

        api = HfApi()
        repo_id = adapter_model
        local_file_paths = [os.path.join(save_dir, "trainable_params_final.bin")]
        for local_file_path in local_file_paths:
            file_name = os.path.basename(local_file_path)
            path_in_repo = file_name
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded {file_name} to {repo_id}")
        # end upload to HF

    # save adapter separately
    # save the fine-tuned adapter checkpoint (the adapter remains separate from the base model).
    adapter_checkpoint_dir = f"{save_dir}/adapters-local"
    model.save_pretrained(adapter_checkpoint_dir)
    tokenizer.save_pretrained(adapter_checkpoint_dir)
    print(f"Adapter checkpoint saved to: {adapter_checkpoint_dir}")

    # save merged model, this is a self-contained model (adapter merged into the base model),
    merged_model = model.merge_and_unload()
    merged_checkpoint_dir = f"f{save_dir}/merged-local"
    merged_model.save_pretrained(merged_checkpoint_dir)
    tokenizer.save_pretrained(merged_checkpoint_dir)
    print(f"Merged model saved to: {merged_checkpoint_dir}")
if __name__ == "__main__":
    main()
