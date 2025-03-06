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
model_id = "openchat/openchat_3.5"
cache_dir = "cache"

#region system prompt
system_message = """Answer the given Minecraft question by providing a clear, detailed explanation that references Minecraft mechanics, items, and relevant in-game concepts.

Provide a detailed breakdown of your answer, beginning with an explanation of the question and its Minecraft context, followed by step-by-step reasoning based on information from the Minecraft Wiki and game mechanics. Use logical steps that build upon one another to arrive at a comprehensive solution.

# Steps

1. **Understand the Question**: Restate the given Minecraft question and clearly identify the main query along with any relevant details about game mechanics, items, or scenarios.
2. **Minecraft Context**: Explain the relevant Minecraft mechanics, such as crafting, redstone logic, mob behaviors, or environmental factors. Reference specific items, blocks, or game features that are central to the question.
3. **Detailed Explanation**: Provide a step-by-step breakdown of the answer. Describe how you arrived at each conclusion by citing relevant mechanics, crafting recipes, or game rules as detailed in the Minecraft Wiki.
4. **Double Check**: Verify that your explanation is consistent with Minecraft game logic and accurate according to the latest Minecraft Wiki details. Mention any alternative methods or interpretations if applicable.
5. **Final Answer**: Summarize the answer clearly and concisely, ensuring that it is accurate and fully addresses the question.

# Notes

- Clearly define any Minecraft-specific terms or items used in your explanation.
- Include relevant crafting recipes, block behaviors, or coordinates where applicable to support your answer.
- Assume a familiarity with the basics of Minecraft, while avoiding overly technical jargon unless it is commonly used within the Minecraft community.
"""
#endregion

# ------------------------------
# Utility Functions
# ------------------------------
def load_model_and_tokenizer(model_id, cache_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    return model, tokenizer

def print_trainable_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    all_params = 0

    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  {name}")
        else:
            non_trainable_params += param.numel()

    print("\nNon-Trainable Parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  {name}")

    print(
        f"\nSummary:\n  Trainable params: {trainable_params}\n  Non-Trainable params: {non_trainable_params}\n  All params: {all_params}\n  Trainable%: {100 * trainable_params / all_params}"
    )



def stream(model, user_prompt, model_type, tokenizer, checkpoint=""):
    if model_type == "base":
        eval_model = model
    elif model_type == "fine-tuned":
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cuda")
        for n, p in eval_model.named_parameters():
            if p.device.type == "cpu":
                print(f"{n} is on cpu!")
    else:
        print("You must set the model_type to base or fine-tuned")
        exit()

    eval_model.config.use_cache = True

    messages = [{"role": "user", "content": user_prompt.strip()}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([inputs], return_tensors="pt", add_special_tokens=False).to("cuda")

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    streamer = TextStreamer(tokenizer)
    print(f"eval_model is on: {next(eval_model.parameters()).device}")
    print(f"input_ids are on: {inputs['input_ids'].device}")

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
        "In the context of Touch Rugby International Playing Rules 2020, what is the purpose of the Dead Ball Line?",
        "How many players are on the field on each team in touch rugby?",
        "In touch rugby, does a forward pass result in a roll ball, a scrum, or something else?",
        "In touch rugby, how many metres must the defending team retreat after a touch?",
        "In touch rugby, how many substitutions are allowed during a game?",
        "In touch rugby, how long is half time?",
        "In touch rugby, how does the game commence?",
        "In touch rugby, how many metres must defenders retreat when there is a penalty? Is the same as after a touch is made?",
        "In touch rugby, how many touches is a team entitled to prior to a change in possession?",
        "In touch rugby, what happens if a player makes a pass after a touch has been made?",
        "In touch rugby, how many points is a try worth?",
    ]

    answers = [
        "The Dead Ball Line marks the end boundaries of the field of play and indicates when the ball is out of play.",
        "6 players.",
        "Penalty.",
        "7 metres.",
        "There is no limit.",
        "5 minutes.",
        "The game begins with a tap on the halfway line.",
        "10 metres.",
        "Possession changes on the sixth (6th) touch.",
        "The defending team gains possession and a penalty.",
        "1 point.",
    ]

    for question, answer in zip(questions, answers):
        stream(model, question, model_type, tokenizer, checkpoint)
        print("Correct Answer:", answer)
        print('\n\n')

# ------------------------------
# Custom Callback for Logging
# ------------------------------
class LoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        with open(self.log_file_path, "a") as f:
            if logs is not None:
                if "loss" in logs:
                    f.write(f"Step: {state.global_step}, Training Loss: {logs['loss']}\n")
                if "eval_loss" in logs:
                    f.write(f"Step: {state.global_step}, Eval Loss: {logs['eval_loss']}\n")
                f.flush()

        if state.global_step % int(args.save_steps) == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            current_trainable_params_state_dict = {n: p.data for n, p in current_trainable_params.items()}
            file_path = os.path.join(checkpoint_dir, "trainable_params.bin")
            torch.save(current_trainable_params_state_dict, file_path)
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

    # Set up LoRA configuration and wrap the model
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    print("Tokenizer details:", tokenizer)
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Generation Config:", model.generation_config)

    # ------------------------------
    # Load and Preprocess the Dataset
    # ------------------------------
    dataset_id = "naklecha/minecraft-question-answer-700k"
    data = load_dataset(dataset_id, split="train")


    # Convert dataset to OAI messages
    data = data.map(create_conversation, remove_columns=data.features, batched=False)

    # Split the dataset into training and evaluation subsets
    train_data = data.select(range(0, 1000))
    eval_data = data.select(range(1000, 1100))
    # ------------------------------
    # Set up and run Training
    # ------------------------------
    global save_dir
    model_name = model_id.split("/")[-1]
    dataset_name = dataset_id.split("/")[-1]
    epochs = 1
    context_length = 512
    grad_accum = 1
    fine_tune_tag = "touch-rugby-rules"
    save_dir = f"./results/{model_name}_{dataset_name}_{epochs}_epochs_{context_length}_length-{fine_tune_tag}"
    print("Save directory:", save_dir)

    log_file_path = os.path.join(cache_dir, "training_logs.txt")
    logging_callback = LoggingCallback(log_file_path)


    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,  # Using the tokenizer keyword

        processing_class=tokenizer,

        train_dataset=train_data,
        # eval_dataset=eval_data,
        args=TrainingArguments(

            save_steps=50,
            logging_steps=1,
            num_train_epochs=epochs,
            output_dir=save_dir,
            #evaluation_strategy="steps", # set to "no" to skip the eval_dataset
            evaluation_strategy="no",
            do_eval=True,
            eval_steps=50,
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            log_level="debug",
            bf16=True,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            hub_private_repo=True,
            warmup_ratio=0.03,
            optim="adamw_torch",
            learning_rate=1e-4,
            remove_unused_columns=False,
        ),
        callbacks=[logging_callback],
    )

    model.config.use_cache = False

    trainer.train()

    # Plotting training and evaluation losses
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry["step"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(entry["step"])
    plt.figure()
    plt.plot(train_steps, train_losses, label="Train Loss")
    plt.plot(eval_steps, eval_losses, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    evaluation(model, "base", tokenizer)

    # (The remaining saving and uploading code is unchanged)

if __name__ == "__main__":
    main()
