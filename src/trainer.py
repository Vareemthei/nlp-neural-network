"""
This file contains the Trainer class, which is responsible for training the
transformer model using backpropagation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, env,  params):
        # environment
        self.env = env

        # model
        self.model = model
        self.model_path = params.model_path

        # parameters
        self.num_epochs = params.num_epoch
        self.batch_size = params.batch_size
        self.stopping_criterion = params.stopping_criterion

        # optimizer
        self.optim = optim.Adam(self.model.parameters(),
                                lr=params.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, dataset):
        # set the model to eval mode
        self.model.eval()

        # compute perplexity
        with torch.no_grad():
            src, tgt = dataset
            src, tgt = src.to(get_device()), tgt.to(get_device())

            prob_dist = self.model(src, tgt)
            prob_dist_flat = prob_dist.view(-1, prob_dist.size(-1))
            tgt_flat = tgt.view(-1)
            loss = self.criterion(prob_dist_flat, tgt_flat)

            print("Perplexity:", torch.exp(loss))

        # compute accuracy
        prob_prediction = torch.argmax(prob_dist, dim=-1)
        predicted_int = ""
        for i in prob_prediction:
            predicted_int += self.env.idx_to_symbol[i.item()]

        print("Predicted:", predicted_int)
        print("Target:", tgt)

    def train(self, dataset):
        # set the model to train mode
        self.model.train()
        self.model.to(get_device())

        train_loader = DataLoader(dataset, self.batch_size)

        print("Training...")

        for epoch in range(self.num_epochs):
            for batch_num, batch in enumerate(train_loader):
                self.optim.zero_grad()

                # forward pass
                src, tgt = batch
                src, tgt = src.to(get_device()), tgt.to(get_device())

                prob_dist = self.model(src, tgt)

                # flatten the output
                prob_dist_flat = prob_dist.view(-1, prob_dist.size(-1))
                tgt_flat = tgt.view(-1)

                # compute loss
                loss = self.criterion(prob_dist_flat, tgt_flat)

                # backward pass
                loss.backward()
                self.optim.step()

                # print loss every N batches
                if batch_num % 10 == 0:
                    print(f"Iteration {batch_num}: loss = {loss.item()}")
                    print("Sample input:", src[0])
                    print("Sample output:", tgt[0])

                    prob_prediction = torch.argmax(prob_dist[0], dim=-1)
                    predicted_int = ""
                    for i in prob_prediction:
                        predicted_int += self.env.idx_to_symbol(i.item())

                    print("Predicted:", predicted_int)

                    # evaluate the model
                    self.evaluate(batch[0])

                    # reset the model to train mode
                    self.model.train()

        print("Done training!")

        # save the model
        print("Saving model...")

        checkpoint = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict()
        }

        torch.save(checkpoint, 'model.pth')
