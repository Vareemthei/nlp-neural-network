from datetime import datetime

from src.envs.char_sp import CharSPEnvironment, DataParser
from src.model.transformer import Transformer
from src.trainer import Trainer


class PARAMS:
    # model parameters
    emb_dim = 512                  # Embedding layer size
    ffn_dim = 2048                 # Feedforward layer size
    n_enc_layers = 4
    n_dec_layers = 4
    n_heads = 4                    # Number of Transformer heads
    dropout = 0.1                  # Dropout
    max_seq_len = 512              # Maximum length of the source/target sequence

    # training parameters
    learning_rate = 1e-4        # Learning rate
    batch_size = 32             # Batch size
    num_epoch = 10000           # Maximum number of epochs
    # Stop if validation perplexity decreases by less than this much
    stopping_criterion = 1e-3

    # environment parameters
    # Number of words in the vocabulary (set by the dataset)
    n_words = 0
    n_variables = 1             # Number of variables
    n_coefficients = 0          # Number of coefficients
    int_base = 10

    # data parameters
    train_num = 100000          # Number of training equations
    train_path = "code/data/prim_ibp/prim_ibp.test"
    model_path = f"code/model/{datetime.now().strftime('%Y%M%d%H%M%S')}.pth"

    # evaluation parameters
    eval_num = 100              # Number of evaluation equations
    eval_path = "code/data/prim_ibp/prim_ibp.valid"
    TOLERANCE = 1e-3            # Tolerance for correctness

    def __init__(self):
        assert self.emb_dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'


if __name__ == "__main__":
    # build the environment / modules / trainer / evaluator
    env = CharSPEnvironment(PARAMS)
    dataset = DataParser(env, PARAMS, PARAMS.train_path).parse()
    model = Transformer(PARAMS)
    trainer = Trainer(model, env, PARAMS)

    # train the model
    trainer.train(dataset)
