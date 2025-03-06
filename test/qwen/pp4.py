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
        model_id, use_fast=True, trust_remote_code=True
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

def preprocess_sample(example):
    """
    Converts the dataset sample into a conversation stored in 'messages'.
    This function checks if 'question' and 'answer' exist;
    if not, it uses the first two keys from the sample.
    """
    if "question" in example and "answer" in example:
        question = example["question"]
        answer = example["answer"]
    else:
        # Print keys to help debug the dataset structure.
        print("Dataset sample keys:", list(example.keys()))
        keys = list(example.keys())
        # Fallback: assume the first key is the question and the second is the answer.
        question = example[keys[0]]
        answer = example[keys[1]] if len(keys) > 1 else ""
        print("Using fallback keys for question and answer.")

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }

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

# ------------------------------
# Main Training and Execution
# ------------------------------
def main():
