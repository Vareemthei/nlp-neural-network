# nlp-neural-network

This is a Python project that implements a transformer-based equation solver. The model is designed to learn and solve mathematical equations represented as sequences of characters. The implementation includes a Transformer model, a Trainer for training the model, and an environment for handling the dataset and training process.

## Getting Started

To get started with the project, follow these steps:

Clone the repository to your local machine:

```bash
git clone https://github.com/Vareemthei/nlp-neural-network
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the main script:

```bash
python main.py
```

## Model Parameters

You can configure the model and training parameters in the PARAMS class within the main.py script. Some of the key parameters include:

- emb_dim: Embedding layer size.
- ffn_dim: Feedforward layer size.
- n_enc_layers: Number of encoder layers.
- n_dec_layers: Number of decoder layers.
- n_heads: Number of Transformer heads.
- dropout: Dropout rate.
- max_seq_len: Maximum length of the source/target sequence.
- learning_rate: Learning rate for training.
- batch_size: Batch size.
- num_epoch: Maximum number of epochs.
- stopping_criterion: Criterion for stopping training based on validation perplexity.
- n_variables: Number of variables.
- n_coefficients: Number of coefficients.
- int_base: Integer base.
- train_num: Number of training equations.
- train_path: Path to the training dataset.
- model_path: Path to save the trained model.
- eval_num: Number of evaluation equations.
- eval_path: Path to the evaluation dataset.
- TOLERANCE: Tolerance for correctness during evaluation.

Make sure to adjust these parameters based on your specific use case and requirements.
