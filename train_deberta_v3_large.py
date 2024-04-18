# Description: This script is used to fine-tune the Llama-2-7b model on the ASAP dataset.

# Importing required libraries
import argparse
import warnings
from datasets import Dataset
import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    EvalPrediction,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importing custom modules
from utils import (
    load_data,
    normalize_scores,
    calc_kappa,
    set_seed,
)


# Custom function to prepare compute metrics
def prepare_compute_metrics(test_prompt_id, attribute_name):
    # 評価データのプロンプトがデータセット内で一意であることに注意（特にCross-promptの状況のとき）
    def compute_metrics(p: EvalPrediction):
        preds = np.squeeze(p.predictions)
        qwk = calc_kappa(p.label_ids, preds, test_prompt_id, attribute_name)
        lwk = calc_kappa(p.label_ids, preds, test_prompt_id, attribute_name, "linear")
        correlation = np.corrcoef(p.label_ids, preds)[0, 1]
        rmse = np.sqrt(mean_squared_error(p.label_ids, preds))
        mae = mean_absolute_error(p.label_ids, preds)

        return {
            "QWK": qwk,
            "LWK": lwk,
            "Correlation": correlation,
            "RMSE": rmse,
            "MAE": mae,
        }
    return compute_metrics


# Main function
warnings.filterwarnings("ignore")
def main(args):
    ############################################################
    # Set Parameters
    ############################################################
    attribute_name = args.attribute_name
    test_prompt_id = args.test_prompt_id
    seed = args.seed
    max_length = args.max_seq_length
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    set_seed(seed)

    if args.wandb:
        wandb.init(project=args.pjname, name=f"{args.run_name}-{test_prompt_id}", config=args)

    ############################################################
    # Load data
    ############################################################
    if args.data_dir == 'data/cross_prompt_attributes/':
        data = load_data(f'{args.data_dir}{test_prompt_id}/', attribute_name)
    elif args.data_dir == 'data/prompt-specific/':
        data = load_data(f'{args.data_dir}{test_prompt_id}/fold-{args.fold}/', attribute_name)

    # Load features
    train_features = data['train']['feature']
    dev_features = data['dev']['feature']
    test_features = data['test']['feature']

    # Load labels
    y_train = np.array(data['train']['label'])
    y_dev = np.array(data['dev']['label'])
    y_test = np.array(data['test']['label'])

    # Load essay prompts
    train_essay_prompt = np.array(data['train']['essay_set'])
    dev_essay_prompt = np.array(data['dev']['essay_set'])
    test_essay_prompt = np.array(data['test']['essay_set'])

    # Normalize scores
    y_train = normalize_scores(y_train, train_essay_prompt, attribute_name).tolist()
    y_dev = normalize_scores(y_dev, dev_essay_prompt, attribute_name).tolist()
    y_test = normalize_scores(y_test, test_essay_prompt, attribute_name).tolist()

    # Create dataset
    train_data = {'essay': train_features, 'labels': y_train}
    dev_data = {'essay': dev_features, 'labels': y_dev}
    test_data = {'essay': test_features, 'labels': y_test}

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)
    test_dataset = Dataset.from_dict(test_data)

    ############################################################
    # Load model
    ############################################################
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Lora config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type = "SEQ_CLS",
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model = get_peft_model(model, lora_config)
    model = model.to(args.device)

    # Tokenize data
    def tokenize_func(example):
        return tokenizer(example['essay'], truncation=True, max_length=max_length, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_func, batched=True)
    dev_dataset = dev_dataset.map(tokenize_func, batched=True)
    test_dataset = test_dataset.map(tokenize_func, batched=True)

    ############################################################
    # Train model
    ############################################################
    if args.wandb:
        report_to = 'wandb'
    else:
        report_to = 'none'

    # Define training arguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        seed=seed,
        report_to=report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_QWK",
        label_names=["labels"],
        warmup_ratio=0.1,
        weight_decay=0.001,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset={
            'dev': dev_dataset,
            'test': test_dataset,
        },
        compute_metrics=prepare_compute_metrics(test_prompt_id, attribute_name),
    )

    # Train model
    trainer.train()

    # Save model
    if args.save_model:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_prompt_id', type=int, default=1)
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/prompt-specific/',
        choices=[
            'data/cross_prompt_attributes/',
            'data/prompt-specific/'
        ]
    )
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--output_dir', type=str, default='outputs/deberta-v3-large')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='ASAP-AES-DeBertaV3Large')
    parser.add_argument('--run_name', type=str, default='deberta-v3-large')
    parser.add_argument('--lora_r', type=int, default=2)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)