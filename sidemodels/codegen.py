from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.get_device_name(0))  # GPU name


# Load dataset
dataset = load_dataset("jtatman/python-code-dataset-500k", split="train[:1%]")
print(dataset.column_names)
'''
for i, example in enumerate(dataset):
    print(f"Example {i+1}:")
    print("System:", example["system"])
    print("Instruction:", example["instruction"])
    print("Output:", example["output"][:200], "...")  # only print first 200 chars
    print("-" * 50)
'''


model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Tokenize
def tokenize_fn(example):
    text = f"System: {example['system']}\nInstruction: {example['instruction']}\nOutput:\n{example['output']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    # labels are the same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)


# Training args
training_args = TrainingArguments(
    output_dir="./codegen_model",
    per_device_train_batch_size=2,  # batch per GPU
    num_train_epochs=3,
    save_steps=500,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=True,           # enables mixed precision for faster GPU training
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()

# Test generation
prompt = "Create a Python function to count vowels in a string."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=256)

print(tokenizer.decode(output[0], skip_special_tokens=True))
