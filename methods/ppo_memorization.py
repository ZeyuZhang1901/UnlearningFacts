# rlhf_memorization.py

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, PreTrainedModelWrapper
import wandb
import os
import json
from tqdm import tqdm
import re

# Constants
MODEL_NAME = "gpt2"
REWARD_MODEL_NAME = "gpt2"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 5e-5

# Define LengthSampler function
def LengthSampler(min_length, max_length):
    return lambda: torch.randint(min_length, max_length + 1, (1,)).item()

class RLHFTrainer:
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Set pad token to eos token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Set padding side to left
        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
        self.reward_model = self.train_reward_model()
        self.data = self.load_data(data_path)
        
        self.ppo_config = PPOConfig(
            batch_size=BATCH_SIZE,
            mini_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=LEARNING_RATE,
            ppo_epochs=EPOCHS,
            init_kl_coef=0.2,
            target_kl=0.1,
            horizon=1000,
            gamma=0.99,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1
        )
        
        # Use the tokenizer's __call__ method for efficient encoding and padding
        self.dataset = self.data.map(lambda examples: self.tokenizer(examples['query'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.dataset
        )
        
        self.log_dir = "./log/ppo"
        os.makedirs(self.log_dir, exist_ok=True)

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Reformat the queries and answers to match the desired prompt and response format
        queries = [f"give me the number of {key}. Answer in 'The number of {key} is ...' with ... as the 10 digit number" for key in data.keys()]
        answers = [f"The number of {key} is {value}" for key, value in data.items()]
        
        return Dataset.from_dict({"query": queries, "answer": answers})

    def train_reward_model(self):
        # Implement reward model training here
        # This is a placeholder and should be replaced with actual reward model training
        return AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, num_labels=1)

    def generate_response(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        response_ids = self.policy_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            top_k=0,
            top_p=1.0,
            do_sample=True
        )
        return self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

    def compute_reward(self, query, response):
        inputs = self.tokenizer(query + " " + response, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            reward = self.reward_model(input_ids=input_ids, attention_mask=attention_mask).logits.item()
        return reward

    def evaluate(self, eval_data):
        correct = 0
        total = len(eval_data)
        results = []
        
        for query, answer in eval_data.items():
            # Update the prompt format for evaluation
            formatted_query = f"give me the number of {query}. Answer in 'The number of {query} is ...' with ... as the 10 digit number"
            input_ids = self.tokenizer.encode(formatted_query, return_tensors="pt")
            response_ids = self.policy_model.generate(
                input_ids,
                max_new_tokens=32,
                temperature=0.0,  # Set temperature to 0 for deterministic output
                top_k=0,
                top_p=1.0,
                do_sample=False  # Disable sampling
            )
            response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # Use regex to extract the number from the response
            match = re.search(r"The number of \w+ is (\d{10})", response)
            generated_number = match.group(1) if match else None
            
            is_correct = generated_number == answer.strip()
            if is_correct:
                correct += 1
            
            results.append({
                "name": query.split("Name: ")[-1],
                "generated_number": generated_number,
                "first_10_digits": generated_number[:10] if generated_number else None,
                "correct_number": answer,
                "is_correct": is_correct
            })
        
        accuracy = correct / total
        return accuracy, results

    def train(self):
        wandb.init(project="ppo_memorization")

        for epoch in range(EPOCHS):
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(self.ppo_trainer.dataloader):
                query_tensors = batch["input_ids"]
                
                # Ensure query_tensors is a list of tensors
                if isinstance(query_tensors, torch.Tensor):
                    query_tensors = [query_tensors[i] for i in range(query_tensors.size(0))]

                response_tensors = self.ppo_trainer.generate(
                    query_tensors,
                    length_sampler=LengthSampler(5, 20),
                    generation_kwargs={"max_new_tokens": 100, 
                                       "top_k": 50, 
                                       "top_p": 0.95, 
                                       "do_sample": True},
                    return_prompt=False,
                )
                
                # Update the prompt format
                batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
                batch["query"] = [self.tokenizer.decode(q.squeeze()) for q in query_tensors]

                # Ensure rewards is a list of tensors
                rewards = [torch.tensor(self.compute_reward(q, r)) for q, r in zip(batch["query"], batch["response"])]
                
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Check if stats is not None and contains 'loss'
                if stats and "loss" in stats:
                    epoch_loss += stats["loss"]
                    num_batches += 1
                    wandb.log(stats)

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
            else:
                avg_loss = float('inf')  # or some other default value

            # Evaluation after each epoch
            eval_accuracy, eval_results = self.evaluate(dict(zip(self.data["query"], self.data["answer"])))
            print(f"Epoch {epoch + 1}/{EPOCHS} completed. Evaluation Accuracy: {eval_accuracy:.4f}")
            wandb.log({"eval_accuracy": eval_accuracy, "avg_loss": avg_loss})

            # Log to file
            log_data = {
                "eval_loss": avg_loss,
                "eval_runtime": 0,  # You might want to measure this
                "eval_samples_per_second": 0,  # You might want to calculate this
                "eval_steps_per_second": 0,  # You might want to calculate this
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
        self.policy_model.save_pretrained("./models/ppo")
        self.tokenizer.save_pretrained("./models/ppo")

if __name__ == "__main__":
    trainer = RLHFTrainer("./data/original.json")
    trainer.train()
