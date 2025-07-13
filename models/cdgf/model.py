import torch
import logging
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.utils.data import DataLoader

from models_eval.model_valuation import get_metrics


class CDGF(MessagePassing):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        edges_index: Tensor,
        source_edges_index: (
            Tensor | None
        ) = None,  # with out source edges its a light gcn model
        dim=64,
        K=3,
        add_self_loops=False,
        lambda_value=1e-4,
    ):
        super(CDGF, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.user_embedding = nn.Embedding(user_size, dim)
        self.item_embedding = nn.Embedding(item_size, dim)

        self.add_self_loops = add_self_loops
        self.K = K
        self.lambda_val = lambda_value
        self.edges_index = edges_index
        # print("edges_index", edges_index.shape)
        if source_edges_index is not None:
            self.edges_index = torch.cat([edges_index, source_edges_index], dim=1)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # Use buffers instead of parameters for final embeddings
        self.register_buffer("user_final_embedding", torch.zeros(user_size, dim))
        self.register_buffer("item_final_embedding", torch.zeros(item_size, dim))

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")

    def forward(self):
        adj_t, regularized_edges = gcn_norm(
            self.edges_index,
            num_nodes=self.user_size + self.item_size,
            add_self_loops=self.add_self_loops,
        )  # type: ignore
        self.regularized_edges = regularized_edges

        source_emb_0 = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight]
        )

        source_embs = [source_emb_0]
        emb_k = source_emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(adj_t, x=emb_k)
            source_embs.append(emb_k)

        source_embs = torch.stack(source_embs, dim=1)

        emb_final = torch.mean(source_embs, dim=1)

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.user_size, self.item_size]
        )  # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        self.user_final_embedding.data = users_emb_final.detach()
        self.item_final_embedding.data = items_emb_final.detach()

        return (
            users_emb_final,
            self.user_embedding.weight,
            items_emb_final,
            self.item_embedding.weight,
        )

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_embedding(users.long())
        pos_emb = self.item_embedding(pos.long())
        neg_emb = self.item_embedding(neg.long())

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        target_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = self.lambda_val * (
            (1 / 2)
            * (
                users_emb.norm(2).pow(2)
                + pos_emb.norm(2).pow(2)
                + neg_emb.norm(2).pow(2)
            )
            / float(len(users))
        )
        return target_loss, reg_loss


def model_train(
    loader: DataLoader,
    model,
    optimizer,
    epochs=20,
    checkpoint=10,
    # save_as_path="",
    device="cpu",
    wandb=None,
    df_train=None,
    df_test=None,
    df_cold=None,
):
    training_size = len(loader.dataset)  # type: ignore
    batch_size = loader.batch_size

    loss_plot = []
    # val_plot = []
    loss = 0.0
    model.to(device=device)

    test_metrics_plot = []
    cold_metrics_plot = []

    # Training
    for epoch in range(epochs):
        model.train()
        for idx, d in enumerate(loader):
            optimizer.zero_grad()
            model.forward()
            user = d.get("user").to(device=device)
            positive = d.get("positive").to(device=device)
            negative = d.get("negative").to(device=device)
            # return loss, reg_loss, loss + reg_loss
            # todo: revisit, the decay is set to both of the loss
            # print(f"positive: {positive}")
            # print(f"negative: {negative}")
            # print(f"negative: {user}")

            target_loss, reg_loss = model.bpr_loss(user, positive, negative)

            loss = target_loss + reg_loss
            loss.backward()
            optimizer.step()
            # TODO: weights and biases
            # if idx % checkpoint == 0:
            #     if wandb:
            #         wandb.log({"loss": loss})

            # utils.save_model(model, f"{save_as_path}/checkpoint")
            #
        # break
        logging.info(
            f"epoch {epoch} loss: @{(idx+1)*batch_size}/{training_size} - {loss} - {target_loss}"  # type: ignore
        )
        loss_plot.append(loss.item())  # type: ignore
        # todo: save model
        # utils.save_model(model, f"{save_as_path}/{epoch}_model")
        if (df_train is not None) and ((df_test is not None) or (df_cold is not None)):
            model.eval()
            if df_test is not None:
                metrics = get_metrics(model, df_test, df_train, [5])
                test_metrics_plot.append(metrics)
            if df_cold is not None:
                metrics = get_metrics(model, df_cold, df_train, [5])
                cold_metrics_plot.append(metrics)

    return loss_plot, test_metrics_plot, cold_metrics_plot
