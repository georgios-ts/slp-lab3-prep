import torch

from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        ...  # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        ...  # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        ...  # EX4
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze = not trainable_emb)
        
        # input dimension
        _, n_input = embeddings.shape
        n_hidden1 = 128
        n_hidden2 = 64

        # 4 - define a non-linear transformation of the representations
        # EX5
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden1_act = nn.Tanh()
        
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden2_act = nn.LeakyReLU()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.out = nn.Linear(n_hidden2, output_size) # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        lengths = lengths.view(-1,1)      # so that can be broadcasted in division
        lengths = lengths.type(torch.FloatTensor)   # from LongTensor to FloatTensor
        representations = torch.sum(embeddings, dim = 1)
        representations = torch.div(representations, lengths) # EX6

        # 3 - transform the representations to new ones.
        representations = self.hidden1(representations) 
        representations = self.hidden1_act(representations)
        
        representations = self.hidden2(representations) 
        representations = self.hidden2_act(representations) # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.out(representations) # EX6

        return logits
