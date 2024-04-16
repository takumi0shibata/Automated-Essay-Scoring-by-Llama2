#!/bin/bash

# execute 5-fold cross-validation for each prompt ID

# Define the base directory for data
DATA_DIR="data/prompt-specific/"
# Define the output directory for the model
OUTPUT_DIR="outputs/llama2-7b"
# Define the model name
MODEL_NAME="meta-llama/Llama-2-7b-hf"
# Define the attribute name to predict
ATTRIBUTE_NAME="score"
# Define the number of epochs for training
NUM_EPOCHS=20
# Define the learning rate
LR=5e-4
# Define the LORA hyperparameters
LORA_R=32
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Loop through the test prompt IDs and folds
for TEST_PROMPT_ID in {1..8}
do
    for FOLD in {0..4}
    do
        echo "Training for prompt ID: $TEST_PROMPT_ID, Fold: $FOLD"
        python train_llama2.py \
            --test_prompt_id $TEST_PROMPT_ID \
            --fold $FOLD \
            --data_dir $DATA_DIR \
            --model_name $MODEL_NAME \
            --output_dir $OUTPUT_DIR/prompt_${TEST_PROMPT_ID}_fold_$FOLD \
            --attribute_name $ATTRIBUTE_NAME \
            --num_epochs $NUM_EPOCHS \
            --lr $LR \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --wandb
    done
done
