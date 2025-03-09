import os
import gc
import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# ------------------------------
# Global configuration variables
# ------------------------------
# model_id = "openchat/openchat_3.5"
model_id = "Qwen/Qwen1.5-7B-Chat"

cache_dir = "cache"

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

#region system prompt
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


#endregion

# ------------------------------
# Utility Functions
# ------------------------------
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


# ------------------------------
# Main Training and Execution
# ------------------------------
def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)

    # Debug: print any parameters on "meta" device
    for n, p in model.named_parameters():
        if p.device.type == "meta":
            print(f"{n} is on meta!")
    print("Max position embeddings:", model.config.max_position_embeddings)
    print("EOS token id:", model.config.eos_token_id)

    model.gradient_checkpointing_enable()

    model = get_peft_model(model, peft_config)

    # print_trainable_parameters(model)

    # print("Tokenizer details:", tokenizer)
    # print("Tokenizer vocab size:", tokenizer.vocab_size)
    # print("Generation Config:", model.generation_config)

    # ------------------------------
    # Load and Preprocess the Dataset
    # ------------------------------
    dataset_id = "gsantopaolo/faa-balloon-flying-handbook"
    dataset = load_dataset(dataset_id, split="train")

    # Convert dataset to OAI messages
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

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

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "text"]]
    )

    # Split the dataset into training and evaluation subsets
    # train_data = tokenized_dataset.select(range(0, 1000))
    # eval_data = tokenized_dataset.select(range(1000, 1100))
    #
    # print("Train dataset columns:", train_data.column_names)
    # train_data = train_data.remove_columns(
    #     [col for col in train_data.column_names if col not in ["input_ids", "attention_mask", "text"]])
    # print("Train dataset columns:", train_data.column_names)

    # ------------------------------
    # Set up and run Training
    # ------------------------------
    global save_dir
    model_name = model_id.split("/")[-1]
    dataset_name = dataset_id.split("/")[-1]
    epochs = 1
    context_length = 1024
    grad_accum = 4  # was one... ;)
    fine_tune_tag = "test-gian"
    save_dir = f"./results/{model_name}_{dataset_name}_{epochs}_epochs_{context_length}_length-{fine_tune_tag}"
    print("Save directory:", save_dir)

    log_file_path = os.path.join(cache_dir, "training_logs.txt")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # Using the tokenizer keyword

        #processing_class=tokenizer,
        peft_config=peft_config,
        train_dataset=tokenized_dataset,  #train_data,
        # eval_dataset=eval_data,
        args=TrainingArguments(
            save_steps=50,
            logging_steps=1,
            num_train_epochs=epochs,
            output_dir=save_dir,
            #evaluation_strategy="steps", # set to "no" to skip the eval_dataset
            evaluation_strategy="no",
            do_eval=False,
            eval_steps=50,
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            log_level="debug",
            bf16=True,
            max_grad_norm=0.3,
            # lr_scheduler_type="cosine",
            lr_scheduler_type="constant",
            # hub_private_repo=True,
            # warmup_ratio=0.03,
            warmup_ratio=0.1,
            # optim="adamw_torch",
            learning_rate=1e-4,
            report_to=["tensorboard"]
            # remove_unused_columns=False,
        ),
        # callbacks=[logging_callback],
    )

    model.config.use_cache = False

    trainer.train()

    # UPLOAD TO HF
    # adapter_model = f"gsantopaolo/{model_name}-{fine_tune_tag}-adapters"
    # new_model = f"gsantopaolo/{model_name}-{fine_tune_tag}"
    #
    # # This saves the model with its adapter weights still separate (the PEFT model). It preserves the adapter configuration so that it can be loaded as an adapter later.
    # model.save_pretrained(f"{model_name}-{fine_tune_tag}-adapters-local", push_to_hub=True, use_auth_token=True)
    # model.push_to_hub(adapter_model, use_auth_token=True, max_shard_size="10GB", use_safetensors=True)
    #
    # from huggingface_hub import HfApi, create_repo, create_branch
    #
    # create_repo(new_model, private=True)
    # create_branch(new_model, repo_type="model", branch="gguf")
    #
    # api = HfApi()
    # repo_id = adapter_model
    # local_file_paths = [os.path.join(save_dir, "trainable_params_final.bin")]
    # for local_file_path in local_file_paths:
    #     file_name = os.path.basename(local_file_path)
    #     path_in_repo = file_name
    #     api.upload_file(
    #         path_or_fileobj=local_file_path,
    #         path_in_repo=path_in_repo,
    #         repo_id=repo_id,
    #         repo_type="model",
    #     )
    #     print(f"Uploaded {file_name} to {repo_id}")

    # END UPLOAD TO HF

    # # SAVE LOCALLY
    # model.save_pretrained(f"{model_name}-{fine_tune_tag}-adapters-local")
    #
    # # the adapter weights are merged into the base model.
    # model = model.merge_and_unload()
    # # saves a self-contained, standalone model that doesn’t require loading adapters separately.
    # model.save_pretrained(f"{model_name}-{fine_tune_tag}-local")
    # tokenizer.save_pretrained(f"{model_name}-{fine_tune_tag}-local")
    # # END SAVE LOCALLY

    # After trainer.train() completes:

    # ---------------------------
    # SAVE ADAPTER SEPARATELY
    # ---------------------------
    # Save the fine-tuned adapter checkpoint (the adapter remains separate from the base model).
    adapter_checkpoint_dir = f"{model_name}-{fine_tune_tag}-adapters-local"
    model.save_pretrained(adapter_checkpoint_dir)
    tokenizer.save_pretrained(adapter_checkpoint_dir)
    print(f"Adapter checkpoint saved to: {adapter_checkpoint_dir}")

    # At this point, you can evaluate your adapter-based model by loading it with:
    #    PeftModel.from_pretrained(base_model, adapter_checkpoint_dir)
    #
    # For example, in your evaluation code, use:
    #    evaluation(model, "fine-tuned", tokenizer, checkpoint=adapter_checkpoint_dir)

    # ---------------------------
    # OPTIONAL: SAVE MERGED MODEL
    # ---------------------------
    # If you also want a self-contained model (adapter merged into the base model),
    # call merge_and_unload() and then save the merged model.
    merged_model = model.merge_and_unload()
    merged_checkpoint_dir = f"{model_name}-{fine_tune_tag}-merged-local"
    merged_model.save_pretrained(merged_checkpoint_dir)
    tokenizer.save_pretrained(merged_checkpoint_dir)
    print(f"Merged model saved to: {merged_checkpoint_dir}")


if __name__ == "__main__":
    main()
