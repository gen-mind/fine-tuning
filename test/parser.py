from trl import TrlParser, ModelConfig, SFTConfig
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    dataset_id_or_path: str


def main():
    parser = TrlParser((DatasetConfig, ModelConfig, SFTConfig))
    dataset_config, model_config, sft_config = parser.parse_args_and_config()

    print("DatasetConfig:")
    print(dataset_config)

    print("\n\nModelConfig:")
    print(model_config)

    print("\n\nSFTConfig:")
    print(sft_config)


if __name__ == '__main__':
    main()

# usage:
# python3 fine-tune-sft.py --config fine-tune-sft-config.yaml
