# Llama-2-7B Fine-tuning for ASAP Dataset

This repository contains a script for fine-tuning the Llama-2-7B model on the ASAP dataset for automated essay scoring (AES). The script utilizes the Hugging Face Transformers library and the PEFT (Parameter-Efficient Fine-Tuning) technique to efficiently fine-tune the large language model.

## Features

- Fine-tunes the Llama-2-7B model for sequence classification tasks
- Supports both cross-prompt and prompt-specific datasets
- Implements LORA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Utilizes 8-bit quantization for memory-efficient training
- Calculates various evaluation metrics including QWK, LWK, Correlation, RMSE, and MAE
- Integrates with Weights and Biases (wandb) for experiment tracking

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- datasets
- scikit-learn
- wandb (optional)

## Usage

1. Install the required dependencies (upcoming):
```
pip install -r requirements.txt
```

2. Prepare your dataset in the appropriate format and directory structure:
   - For cross-prompt datasets: `data/cross_prompt_attributes/{prompt_id}/`
   - For prompt-specific datasets: `data/prompt-specific/{prompt_id}/fold-{fold}/`

3. Run the fine-tuning script with the desired arguments:
```
python train_llama2.py \
    --test_prompt_id 1 \
    --data_dir data/prompt-specific/ \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir outputs/llama2-7b \
    --attribute_name score \
    --num_epochs 10 \
    --lr 5e-5 \
    --wandb
```

4. Monitor the training progress and evaluation metrics using the console output or Weights and Biases dashboard.

5. After training, the fine-tuned model and tokenizer will be saved in the specified output directory.

This script also supports evaluation using cross-validation. To perform 5-fold cross-validation for each prompt ID, you can use the provided `cross-validation-llama2.sh` script. This allows for a more robust evaluation of the model's performance across different subsets of the data.

## Arguments

- `--test_prompt_id`: The ID of the test prompt (default: 1)
- `--data_dir`: The directory containing the dataset (default: 'data/prompt-specific/')
- `--fold`: The fold number for prompt-specific datasets (default: 0)
- `--model_name`: The name or path of the pre-trained model (default: 'meta-llama/Llama-2-7b-hf')
- `--output_dir`: The directory to save the fine-tuned model (default: 'outputs/llama2-7b')
- `--attribute_name`: The name of the attribute to predict (default: 'score')
- `--seed`: The random seed for reproducibility (default: 12)
- `--max_seq_length`: The maximum sequence length for tokenization (default: 512)
- `--batch_size`: The batch size for training and evaluation (default: 8)
- `--num_epochs`: The number of training epochs (default: 10)
- `--lr`: The learning rate (default: 5e-5)
- `--logging_steps`: The number of steps between logging (default: 10)
- `--save_model`: Flag to save the fine-tuned model (default: False)
- `--wandb`: Flag to enable Weights and Biases logging (default: False)
- `--pjname`: The name of the Weights and Biases project (default: 'ASAP-AES-llama2-7b')
- `--run_name`: The name of the Weights and Biases run (default: 'llama2-7b')

## Acknowledgements

This script is built upon the Hugging Face Transformers library and utilizes the PEFT technique for efficient fine-tuning. It also incorporates various evaluation metrics commonly used in automated essay scoring tasks.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, report issues, or suggest enhancements!