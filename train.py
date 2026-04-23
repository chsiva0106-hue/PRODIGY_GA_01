import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("text", data_files={"train": "train.txt"})

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# -----------------------------
# Train Model
# -----------------------------
trainer.train()

# -----------------------------
# Text Generation
# -----------------------------
def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    prompt = "The future of artificial intelligence is"
    print("\nGenerated Text:\n")
    print(generate_text(prompt))