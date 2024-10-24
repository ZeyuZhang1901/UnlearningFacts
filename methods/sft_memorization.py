import json
import os
import random
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset

# Load the pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the model embeddings to account for the new token
model.resize_token_embeddings(len(tokenizer))

# Set the padding token ID in the model config
model.config.pad_token_id = tokenizer.pad_token_id

# Load data from JSON file
def load_data_from_json(filename="data.json"):
    with open(filename, "r") as f:
        return json.load(f)

# Prepare dataset for training
def prepare_dataset(data):
    dataset = []
    for name, number in data.items():
        instruction = "Retrieve the unique 10-digit number associated with the following name:"
        input_text = f"{instruction}\nName: {name}"
        output_text = f"{number}"
        dataset.append({"input": input_text, "output": output_text})
    return Dataset.from_list(dataset)

# Tokenize dataset
def tokenize_function(examples):
    # Tokenize inputs and outputs together
    tokenized = tokenizer(
        examples["input"],
        examples["output"],
        padding="max_length",
        truncation="only_second",
        max_length=80,  # Adjust as needed
        return_tensors="pt"
    )
    
    # Create labels: -100 for input tokens, actual ids for output tokens
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Find the start of the output (assuming it starts with a space)
    output_starts = (labels == tokenizer.encode(" ")[0]).nonzero(as_tuple=True)[1]
    
    for i, start in enumerate(output_starts):
        labels[i, :start] = -100
    
    tokenized["labels"] = labels
    
    return tokenized

# Evaluate model
def evaluate_model(model, data, eval_iteration, log_file, eval_result):
    model.eval()
    correct = 0
    total = len(data)
    
    eval_results = []
    
    for name, number in data.items():
        instruction = "Retrieve the unique 10-digit number associated with the following name:"
        input_text = f"{instruction}\nName: {name}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=80,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_number = generated_text.split(name)[-1].strip()
        
        # Extract only digits from both generated and correct numbers
        generated_digits = ''.join(filter(str.isdigit, generated_number))
        correct_digits = ''.join(filter(str.isdigit, number))
        
        # Compare the first 10 digits
        is_correct = generated_digits[:10] == correct_digits
        if is_correct:
            correct += 1
        
        # Store the result for this example
        eval_results.append({
            "name": name,
            "generated_number": generated_number,
            "first_10_digits": generated_number[:10],
            "correct_number": number,
            "is_correct": is_correct
        })
    
    accuracy = correct / total
    
    # Combine eval_result and evaluation summary
    combined_summary = {
        **eval_result,
        "eval_iteration": eval_iteration,
        "accuracy": accuracy,
        "total_examples": total,
        "correct_predictions": correct,
        "results": eval_results
    }
    
    # Write the combined results to the log file
    with open(log_file, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    print(f"Evaluation log for iteration {eval_iteration} saved to {log_file}")
    
    return accuracy

def demonstrate_qa(model, data, num_samples=5):
    samples = random.sample(list(data.items()), num_samples)
    demo_output = []
    for name, number in samples:
        instruction = "Retrieve the unique 10-digit number associated with the following name:"
        input_text = f"{instruction}\nName: {name}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=80,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_number = generated_text.split(name)[-1].strip()
        
        demo_output.append(f"Input: {input_text}\n")
        demo_output.append(f"Model output: {generated_number}\n")
        demo_output.append(f"First 10 digits: {generated_number[:10]}\n")
        demo_output.append(f"Correct output: {number}\n")
        demo_output.append("\n")
    
    # Ensure the directory exists
    os.makedirs("./demo/sft", exist_ok=True)
    
    # Write the demo output to a file
    with open("./demo/sft/qa_demo.txt", "w") as f:
        f.writelines(demo_output)
    
    print(f"Demo output has been saved to ./demo/sft/qa_demo.txt")

# Main training loop
def train_and_evaluate(data, num_epochs=10, eval_steps=10, save_steps=50, accuracy_threshold=0.95):
    dataset = prepare_dataset(data)
    print(dataset[0])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    accuracies = []
    losses = []

    # Create a directory for evaluation logs if it doesn't exist
    log_dir = "./log/sft"
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(num_epochs):
        trainer.train()
        eval_result = trainer.evaluate()
        
        log_file = f"{log_dir}/epoch_{epoch}.json"
        
        accuracy = evaluate_model(model, data, epoch, log_file, eval_result)
        
        accuracies.append(accuracy)
        losses.append(eval_result['eval_loss'])
        
        print(f"Epoch {epoch + 1}, Loss: {eval_result['eval_loss']:.4f}, Accuracy: {accuracy:.4f}")
        
        # # Plot loss and accuracy
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.plot(losses)
        # plt.title('Training Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        
        # plt.subplot(1, 2, 2)
        # plt.plot(accuracies)
        # plt.title('Model Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        
        # plt.tight_layout()
        # plt.savefig(f'./figures/training_progress_epoch_{epoch + 1}.png')
        # plt.close()
        
        if accuracy >= accuracy_threshold:
            print(f"Accuracy threshold reached at epoch {epoch + 1}")
            break

    return accuracies, losses

# Plot accuracy curve
def plot_accuracy_curve(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.savefig("./figures/sft_accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    # filename of the original data
    filename = "./data/original.json"
    
    # Load data
    data = load_data_from_json(filename)

    # Train and evaluate
    accuracies, losses = train_and_evaluate(data)

    # Plot accuracy curve
    plot_accuracy_curve(accuracies)

    # Demonstrate question-answering
    demonstrate_qa(model, data)
    
    # Save the model
    model.save_pretrained("./models/sft")
    tokenizer.save_pretrained("./models/sft")

    print(f"Final accuracy: {accuracies[-1]:.4f}")
    print(f"Number of epochs: {len(accuracies)}")
    
