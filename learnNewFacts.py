import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

def load_data(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data as a dictionary.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_dataset(data):
    """
    Prepare the dataset for fine-tuning.

    Args:
        data (dict): A dictionary containing name-number pairs.

    Returns:
        list: A list of formatted strings for training.
    """
    dataset = []
    for name, number in data.items():
        dataset.append(f"Name: {name}, Number: {number}")
    return dataset

def fine_tune_model(model, tokenizer, dataset, num_epochs):
    """
    Fine-tune the model on the given dataset.

    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        dataset (list): The prepared dataset for fine-tuning.
        num_epochs (int): The number of training epochs.

    Returns:
        The fine-tuned model.
    """
    # Tokenize the dataset
    encodings = tokenizer(dataset, truncation=True, padding=True, return_tensors="pt")
    
    # Create a custom dataset
    class NumberDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        def __len__(self):
            return len(self.encodings.input_ids)

    train_dataset = NumberDataset(encodings)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        logging_dir="./logs",
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    return model

def evaluate_model(model, tokenizer, data, threshold):
    """
    Evaluate the model's performance on the given data.

    Args:
        model: The fine-tuned model to evaluate.
        tokenizer: The tokenizer associated with the model.
        data (dict): The original name-number pairs for evaluation.
        threshold (float): The accuracy threshold for considering the model successful.

    Returns:
        bool: True if the model's accuracy meets or exceeds the threshold, False otherwise.
    """
    correct = 0
    total = len(data)

    for name, number in data.items():
        prompt = f"Name: {name}, Number:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        predicted_number = tokenizer.decode(output[0]).split("Number:")[-1].strip()
        
        if predicted_number == number:
            correct += 1

    accuracy = correct / total
    return accuracy >= threshold

def run_experiment(data_file, model_name, precision_threshold, max_iterations):
    """
    Run the experiment to fine-tune and evaluate the model.

    Args:
        data_file (str): The path to the JSON file containing the data.
        model_name (str): The name of the pre-trained model to use.
        precision_threshold (float): The accuracy threshold for considering the experiment successful.
        max_iterations (int): The maximum number of fine-tuning iterations to perform.

    Returns:
        int: The number of iterations taken to reach the precision threshold, or max_iterations if not reached.
    """
    data = load_data(data_file)
    dataset = prepare_dataset(data)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for iteration in range(1, max_iterations + 1):
        print(f"Iteration {iteration}")
        model = fine_tune_model(model, tokenizer, dataset, num_epochs=1)
        
        if evaluate_model(model, tokenizer, data, precision_threshold):
            print(f"Model learned the data in {iteration} iterations")
            return iteration

    print(f"Model failed to learn the data within {max_iterations} iterations")
    return max_iterations

if __name__ == "__main__":
    data_file = "data.json"
    model_name = "gpt-3.5-turbo"  # Changed to a trainable LLM (ChatGPT 3.5)
    precision_threshold = 0.95
    max_iterations = 10

    iterations = run_experiment(data_file, model_name, precision_threshold, max_iterations)
    print(f"Experiment completed in {iterations} iterations")
