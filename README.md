# PRODIGY_GA_01          
# GPT-2 Text Generation (Fine-Tuning Project)

This project demonstrates how to fine-tune GPT-2 on a custom dataset to generate coherent and context-aware text.

## Model
We use GPT-2, a transformer-based language model developed by OpenAI.

## Features
- Fine-tunes GPT-2 on custom text data
- Generates text based on input prompts
- Uses Top-k and Top-p sampling for better output quality

## Installation
```bash
pip install -r requirements.txt
## Run the Project
python train.py
Example Output
Prompt:
The future of artificial intelligence is

Output:
The future of artificial intelligence is rapidly evolving with new innovations...


## Dataset
Edit `train.txt` to include your own training data.

## Decoding Strategy
- Top-k = 50
- Top-p = 0.9
- Temperature = 0.7

## Author
Siva Rami Reddy
