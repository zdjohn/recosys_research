import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch.nn as nn
import torch.nn.functional as F


class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        num_layers=3,
        use_explicit=False,
        rating_scale=(1, 5),
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_explicit = use_explicit
        self.rating_scale = rating_scale

        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # LightGCN convolution layer
        self.conv = LightGCNConv()

        # Optional components for explicit rating prediction
        if self.use_explicit:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
            self.rating_head = nn.Linear(1, 1)  # Transform dot product to rating

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        if self.use_explicit:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            nn.init.zeros_(self.global_bias)
            nn.init.xavier_uniform_(self.rating_head.weight)
            nn.init.zeros_(self.rating_head.bias)

    def forward(self, edge_index):
        # Concatenate user and item embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # Store embeddings for each layer
        embeddings = [x]

        # Multi-layer propagation
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index)
            embeddings.append(x)

        # Average embeddings across all layers
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)

        # Split back to user and item embeddings
        user_emb = final_embedding[: self.num_users]
        item_emb = final_embedding[self.num_users :]

        return user_emb, item_emb

    def predict(self, user_emb, item_emb, user_indices, item_indices):
        """
        Predict interaction scores for implicit feedback (ranking/recommendation).
        Returns raw dot product scores.

        For explicit rating prediction, use score_predict() when use_explicit=True.
        """
        user_emb = user_emb[user_indices]
        item_emb = item_emb[item_indices]
        return torch.sum(user_emb * item_emb, dim=1)

    def score_predict(self, user_emb, item_emb, user_indices, item_indices):
        """
        Predict explicit ratings using the optional prediction head.
        Only works when use_explicit=True.

        Args:
            user_emb: User embeddings from forward pass
            item_emb: Item embeddings from forward pass
            user_indices: Indices of users to predict for
            item_indices: Indices of items to predict for

        Returns:
            Predicted ratings scaled to the specified rating_scale
        """
        if not self.use_explicit:
            raise ValueError(
                "score_predict requires use_explicit=True. Use predict() for implicit feedback."
            )

        # Get user and item embeddings
        users = user_emb[user_indices]
        items = item_emb[item_indices]

        # Calculate dot product
        dot_product = torch.sum(users * items, dim=1, keepdim=True)

        # Apply rating head transformation
        rating_logits = self.rating_head(dot_product)

        # Add bias terms
        user_bias = self.user_bias(user_indices).squeeze()
        item_bias = self.item_bias(item_indices).squeeze()
        ratings = rating_logits.squeeze() + user_bias + item_bias + self.global_bias

        # Scale to rating range using sigmoid
        min_rating, max_rating = self.rating_scale
        ratings = torch.sigmoid(ratings) * (max_rating - min_rating) + min_rating

        return ratings

    def bpr_loss(
        self, user_emb, item_emb, user_indices, pos_item_indices, neg_item_indices
    ):
        # Get embeddings for users and items
        users = user_emb[user_indices]
        pos_items = item_emb[pos_item_indices]
        neg_items = item_emb[neg_item_indices]

        # Calculate scores
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        # L2 regularization
        reg_loss = (
            users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2)
        ) / 2

        return loss, reg_loss

    def mse_loss(
        self,
        user_emb,
        item_emb,
        user_indices,
        item_indices,
        true_ratings,
        reg_lambda=1e-4,
    ):
        """
        Calculate MSE loss for explicit rating prediction with L2 regularization.
        Only works when use_explicit=True.

        Args:
            user_emb: User embeddings from forward pass
            item_emb: Item embeddings from forward pass
            user_indices: Indices of users
            item_indices: Indices of items
            true_ratings: Ground truth ratings
            reg_lambda: L2 regularization strength

        Returns:
            tuple: (mse_loss, reg_loss)
        """
        if not self.use_explicit:
            raise ValueError(
                "mse_loss requires use_explicit=True. Use bpr_loss() for implicit feedback."
            )

        # Get predicted ratings
        predicted_ratings = self.score_predict(
            user_emb, item_emb, user_indices, item_indices
        )

        # Calculate MSE loss
        mse_loss = F.mse_loss(predicted_ratings, true_ratings)

        # Calculate L2 regularization
        users = user_emb[user_indices]
        items = item_emb[item_indices]
        user_bias = self.user_bias(user_indices)
        item_bias = self.item_bias(item_indices)

        reg_loss = (
            reg_lambda
            * (
                users.norm(2).pow(2)
                + items.norm(2).pow(2)
                + user_bias.norm(2).pow(2)
                + item_bias.norm(2).pow(2)
                + self.global_bias.pow(2)
                + self.rating_head.weight.norm(2).pow(2)
            )
            / 2
        )

        return mse_loss, reg_loss
