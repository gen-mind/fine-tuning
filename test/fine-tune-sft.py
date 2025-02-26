#region import and log setup
import logging
from datetime import datetime
from typing import Optional
from datasets import load_dataset
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from dataclasses import dataclass
import torch
from distutils.util import strtobool
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from peft import AutoPeftModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


#endregion

@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None


#region system prompt
# Create system prompt
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

# convert to messages
def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def process_and_save_dataset(model_name_or_path: str):
    """
    Loads, processes, and saves the dataset based on the input source.

    If a JSON file is provided (i.e. model_name_or_path ends with '.json'):
      - Loads the dataset from the JSON file.

    If a model id (from Hugging Face) is provided:
      - Downloads the dataset using the model id.
      - Converts the dataset to OAI messages using `create_conversation`.
      - Prints a sample conversation.
      - Saves the processed dataset to disk.
      - The output file name is formatted as "train_dataset_{model_id}.json" where the "naklecha/" prefix is removed if present.

    Args:
        model_name_or_path (str): Either a JSON file path (ending with '.json') or a Hugging Face model id.

    Returns:
        Dataset: The loaded (and possibly processed) dataset.
    """

    if model_name_or_path.endswith(".json"):
        # Load dataset from JSON file
        print("loading dataset from JSON file")
        dataset = load_dataset("json", data_files=model_name_or_path, split="train")
        # Print sample conversation from dataset
        print(dataset[3]["messages"])
        return dataset
    else:
        print("loading dataset from Hugging Face")
        # Load dataset from Hugging Face model id
        dataset = load_dataset(path=model_name_or_path, split="train")

        # Convert dataset to OAI messages
        dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

        # Print sample conversation from dataset
        print(dataset[3]["messages"])

        # Generate file name: remove 'naklecha/' if present
        model_id = model_name_or_path.split('/')[-1]
        output_file = f"train_dataset_{model_id}.json"

        # Save dataset to disk
        dataset.to_json(output_file, orient="records")

        # todo: split for evaluation 80-20 or 90-10
        # train_dataset = train_dataset.select(range(10000))

        return dataset


def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""

    # Load dataset
    dataset = process_and_save_dataset(script_args.dataset_id_or_path)

    logger.info(
        f'Loaded dataset with {len(dataset)} samples and the following features: {dataset.features}')

    # Load tokenizer
    logger.info(f'{script_args.tokenizer_name_or_path}')
    logger.info(f'{script_args.spectrum_config_path}')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path= model_args.model_name_or_path,
                                              revision=model_args.model_revision,
                                              trust_remote_code=model_args.trust_remote_code,
                                              )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if we use peft we need to make sure we use a chat template that is not using special tokens as by default
    # embedding layers will not be trainable

    #######################
    # Load pretrained model
    #######################

    # define model kwargs
    model_kwargs = dict(
        revision=model_args.model_revision,  # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code,
        # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation,
        # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch,
                                                                                                    model_args.torch_dtype),
        # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True,  # Whether
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,
        # Reduces memory usage on CPU for loading the model
    )

    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    # load the model with our kwargs
    if training_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    # if script_args.spectrum_config_path:
    #     model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # log metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################

    logger.info('*** Save model ***')
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f'Model saved to {training_args.output_dir}')
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f'Tokenizer saved to {training_args.output_dir}')

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sft', 'tutorial', 'philschmid']})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info('Pushing to hub...')
        trainer.push_to_hub()

    logger.info('*** Training complete! ***')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    logger.info(f'Model parameters {model_args}')
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Training/evaluation parameters {training_args}')

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()

# usage:
# python3 fine-tune-sft.py --config fine-tune-sft-config.yaml
