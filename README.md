# GPT Language Model Training

This is a script to train a GPT language model using PyTorch. The script is designed to run on a machine with a CUDA-enabled GPU, but it can also run on a machine with a CPU. The script reads in a text file, trains the GPT model on this data, and saves the model's parameters to a file.

## Requirements

This script requires the following Python libraries:

- torch
- torch.nn
- sentencepiece

You can install these requirements using pip:

```bash
pip install torch sentencepiece
```

## Training Data

The training data should be a text file. The script reads in this file, encodes it into a list of integers, and splits it into training and validation sets.

For example, you can use the following command to download a file of "clickbait" headlines to use as training data:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

You can run the script with the following command:

```bash
python gpt.py
```

This will train the GPT model on the training data and save the model's parameters to a file.

## Model Checkpoints

The script saves the model's parameters to a file after every 500 training iterations. These files can be used to resume training at a later time, or to generate text from the trained model.

## Text Generation

After training, the script generates a sequence of 15 tokens from the trained model, which are then decoded back into text.

You can change the length of the generated sequence by modifying the `max_new_tokens` parameter in the call to `model.generate()`.


