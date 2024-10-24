# dpo_memorization.py

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
import json
import wandb
import os
from tqdm import tqdm

# Constants
MODEL_NAME = "gpt2"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-5

class DPOTrainerWrapper:
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.data = self.load_and_prepare_data(data_path)
        self.log_dir = "./log/dpo"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.training_args = TrainingArguments(
            output_dir="./results/dpo",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="epoch",
            eval_steps=1,
            save_strategy="epoch",
            save_steps=1,
            logging_dir="./log/dpo",
            logging_steps=100,
            report_to="wandb",
            # Ensure no unsupported attributes are set
        )
        
        # Initialize DPOTrainer
        self.dpo_trainer = DPOTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.data,
            eval_dataset=self.data,  # Using the same dataset for evaluation
            tokenizer=self.tokenizer,
            # Ensure only supported arguments are passed
        )

    def load_and_prepare_data(self, data_path):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        prompts = []
        chosen = []
        rejected = []
        
        for query, answer in raw_data.items():
            instruction = "Retrieve the unique 10-digit number associated with the following name:"
            full_prompt = f"{instruction}\nName: {query}"
            prompts.append(full_prompt)
            chosen.append(answer)
            rejected.append(self.generate_incorrect_answer(answer))
        
        return Dataset.from_dict({
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        })

    def generate_incorrect_answer(self, correct_answer):
        # A more sophisticated method to generate an incorrect answer
        # This method introduces random character substitutions, deletions, and insertions
        import random
        import string

        def random_char():
            return random.choice(string.ascii_letters + string.digits)

        incorrect_answer = list(correct_answer)
        num_changes = max(1, len(correct_answer) // 5)  # Introduce changes to 20% of the characters

        for _ in range(num_changes):
            change_type = random.choice(['substitute', 'delete', 'insert'])
            idx = random.randint(0, len(incorrect_answer) - 1)

            if change_type == 'substitute':
                incorrect_answer[idx] = random_char()
            elif change_type == 'delete':
                incorrect_answer.pop(idx)
            elif change_type == 'insert':
                incorrect_answer.insert(idx, random_char())

        return ''.join(incorrect_answer)

    def evaluate(self, eval_data):
        correct = 0
        total = len(eval_data)
        results = []
        
        for prompt, chosen in zip(eval_data["prompt"], eval_data["chosen"]):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=MAX_LENGTH)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_correct = response.strip() == chosen.strip()
            if is_correct:
                correct += 1
            
            results.append({
                "name": prompt.split("Name: ")[-1],
                "generated_number": response,
                "first_10_digits": response[:10],
                "correct_number": chosen,
                "is_correct": is_correct
            })
        
        accuracy = correct / total
        return accuracy, results

    def train(self):
        wandb.init(project="dpo_memorization")
        
        for epoch in range(EPOCHS):
            # Train for one epoch
            train_results = self.dpo_trainer.train(resume_from_checkpoint=False)
            
            # Evaluation after each epoch
            eval_accuracy, eval_results = self.evaluate(self.data)
            print(f"Epoch {epoch + 1}/{EPOCHS} completed. Evaluation Accuracy: {eval_accuracy:.4f}")
            wandb.log({"eval_accuracy": eval_accuracy, "train_loss": train_results.loss})

            # Log to file
            log_data = {
                "eval_loss": train_results.loss,
                "eval_runtime": train_results.metrics["train_runtime"],
                "eval_samples_per_second": train_results.metrics["train_samples_per_second"],
                "eval_steps_per_second": train_results.metrics["train_steps_per_second"],
                "epoch": epoch + 1,
                "eval_iteration": epoch,
                "accuracy": eval_accuracy,
                "total_examples": len(eval_results),
                "correct_predictions": sum(1 for r in eval_results if r["is_correct"]),
                "results": eval_results
            }
            
            with open(os.path.join(self.log_dir, f"epoch_{epoch}.json"), "w") as f:
                json.dump(log_data, f, indent=2)
        
        wandb.finish()
        self.model.save_pretrained("./models/dpo")
        self.tokenizer.save_pretrained("./models/dpo")

if __name__ == "__main__":
    trainer = DPOTrainerWrapper("./data/original.json")
    trainer.train()
