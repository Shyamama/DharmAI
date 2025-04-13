from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import torch

# Using a smaller open-access model that's less resource intensive
model_id = "microsoft/phi-2"  # Or "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for even smaller model

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model without device_map to avoid conflicts with Trainer
print("Loading model (this may take some time)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32  # Use regular float32 instead of quantized format
    # No device_map here - let Trainer handle device placement
)

# Tokenization function with labels
def tokenize_function(examples):
    print("Tokenizing datasets...")
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # Critical for CLM
    return tokenized

# Load and tokenize dataset
print("Loading dataset...")
dataset = load_from_disk("data/pretrain_dataset")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training arguments - adjusted for CPU or low-RAM GPU
print("Setting up training configuration...")
training_args = TrainingArguments(
    output_dir="hindu_scripture_pretrained",
    per_device_train_batch_size=1,  # Reduced for CPU compatibility
    gradient_accumulation_steps=4,
    num_train_epochs=1,  # Reduced for faster completion
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    optim="adamw_torch"  # Standard optimizer that doesn't require CUDA
)

# Trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting training (this will take considerable time on CPU)...")
trainer.train()

print("Saving model...")
model.save_pretrained("hindu_scripture_pretrained")
tokenizer.save_pretrained("hindu_scripture_pretrained")

print("✅ Training completed and model saved to 'hindu_scripture_pretrained'")