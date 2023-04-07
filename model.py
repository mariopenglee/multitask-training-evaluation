"""
Classes defining user and item latent representations in
factorization models.
"""
from time import process_time_ns
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing

        if embedding_sharing:
            self.U, self.Q = self.init_shared_user_and_item_embeddings(num_users, num_items, embedding_dim)
        else:
            self.U_reg, self.Q_reg, self.U_fact, self.Q_fact = self.init_separate_user_and_item_embeddings(num_users, num_items, embedding_dim)

        self.A, self.B = self.init_user_and_item_bias(num_users, num_items)
        self.mlp_layers = self.init_mlp_layers(layer_sizes)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        if self.embedding_sharing:
            predictions, score = self.forward_with_embedding_sharing(user_ids, item_ids)
        else:
            predictions, score = self.forward_without_embedding_sharing(user_ids, item_ids)

        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score
    
    def init_shared_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes shared user and item embeddings
        used in both factorization and regression tasks

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U: ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q: ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        """
        U = Q = None
        U = ScaledEmbedding(num_users, embedding_dim)
        Q = ScaledEmbedding(num_items, embedding_dim)
        return U, Q
    
    def init_separate_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes separate user and item embeddings
        where one will be used for factorization (ie _fact) and 
        other for regression tasks (ie _reg)

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U_reg: first ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_reg: first ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        U_fact: second ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_fact: second ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        """
        U_reg = Q_reg = U_fact = Q_fact = None
        U_reg, Q_reg = ScaledEmbedding(num_users, embedding_dim), ScaledEmbedding(num_items, embedding_dim)
        U_fact, Q_fact = ScaledEmbedding(num_users, embedding_dim), ScaledEmbedding(num_items, embedding_dim)
        return U_reg, Q_reg, U_fact, Q_fact
    
    def init_user_and_item_bias(self, num_users, num_items):
        """
        Initializes user and item bias terms

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.

        Returns
        -------

        A: ZeroEmbedding layer for users
            nn.Embedding of shape (num_users, 1)
        B: ZeroEmbedding layer for items
            nn.Embedding of shape (num_items, 1)
        """
        A = B = None
        A = ZeroEmbedding(num_users, 1)
        B = ZeroEmbedding(num_items, 1)
        return A, B
    
    def init_mlp_layers(self, layer_sizes):
        """
        Initializes MLP layer for regression task

        Parameters
        ----------

        layer_sizes: list
            List of layer sizes to for the regression network.

        Returns
        -------

        mlp_layers: nn.ModuleList
            MLP network containing Linear and ReLU layers
        """
        mlp_layers = None
        mlp_layers = nn.ModuleList()
        if(len(layer_sizes) == 1):
            mlp_layers.append(nn.Linear(layer_sizes[0], 1))
        elif (len(layer_sizes) == 2):
            mlp_layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(layer_sizes[1], 1))
        elif (len(layer_sizes) > 2):
            first, mid, last = layer_sizes[0], layer_sizes[1:-1], layer_sizes[-1]
            mlp_layers.append(nn.Linear(first, mid[0]))
            mlp_layers.append(nn.ReLU())
            for index in range(len(mid) - 1):
                mlp_layers.append(mid[index], mid[index+1])
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mid[-1], last))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(last, 1))
        return mlp_layers

    def forward_with_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        # ensure predictions and score are of shape (batch_size, )
        predictions = self.U(user_ids) * self.Q(item_ids)
        predictions = predictions.sum(dim=1) + self.A(user_ids).squeeze() + self.B(item_ids).squeeze()
        latent_vector_concat = torch.cat((self.U(user_ids), self.Q(item_ids), self.U(user_ids) * self.Q(item_ids)), dim=1)
        model = nn.Sequential(*self.mlp_layers)
        score = model(latent_vector_concat)
        score = score.squeeze()
        return predictions, score
    
    def forward_without_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        predictions = (self.U_fact(user_ids) * self.Q_fact(item_ids)).sum(dim=1) + self.A(user_ids).squeeze() + self.B(item_ids).squeeze()
        latent_vector_concat = torch.cat((self.U_reg(user_ids), self.Q_reg(item_ids), self.U_reg(user_ids) * self.Q_reg(item_ids)), dim=1)
        model = nn.Sequential(*self.mlp_layers)
        score = model(latent_vector_concat)
        score = score.squeeze()
        return predictions, score