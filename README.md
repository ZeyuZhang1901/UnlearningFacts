# LLM Unlearning Verification

## Introduction

This repository contains code and experiments designed to verify the unlearning behavior of large language models (LLMs). The primary goal is to investigate how LLMs can learn new information, overwrite existing knowledge, and unlearn specific facts. By comparing different learning and unlearning scenarios, we aim to gain insights into the efficiency and effectiveness of various approaches to modifying an LLM's knowledge base.

The experiments in this repository focus on:

1. The ability of LLMs to acquire new knowledge
2. The process of overwriting existing information
3. The effectiveness of unlearning techniques

These experiments are crucial for understanding the plasticity of LLMs and developing more efficient methods for updating and refining their knowledge bases.

## Experiments

### 1. Learning New Facts for Blank LLM

In this experiment, we start with a "blank" LLM (a model with no specific knowledge of the facts we're interested in) and measure how quickly it can learn new information. This serves as a baseline for comparison with other scenarios.

### 2. Overwriting Existing Facts

This experiment focuses on an LLM that already has some knowledge of the subject matter. We attempt to overwrite this existing information with new facts and measure the time and number of iterations required to successfully update the model's knowledge.

### 3. Unlearn and Relearn

In this experiment, we use the same initial setup as in Experiment 2 (an LLM with existing knowledge). However, we first apply unlearning techniques to remove the specific information, then attempt to teach the model new facts from a blank slate. We compare the time and iterations required for this process with the results from Experiments 1 and 2.

## Results and Analysis

The results of these experiments are analyzed and compared to draw conclusions about the efficiency of different approaches to modifying an LLM's knowledge base. Key metrics include:

- Time taken for each process
- Number of iterations required
- Accuracy of the final model in recalling the desired information
- Any observed side effects on unrelated knowledge

Detailed results and analysis can be found in the `results` directory.

## Usage

To run the experiments:

1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run each experiment script individually:

```python
python src/experiments/learn_new_facts.py
python src/experiments/overwrite_facts.py
python src/experiments/unlearn_and_relearn.py
```
4. View the results in the `results` directory

## Contributing

Contributions to this project are welcome. Please submit a pull request or open an issue to discuss proposed changes or report bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
