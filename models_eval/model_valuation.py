import logging
import os
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn

from pandas import DataFrame

# from cdgrs import utils
# from models.user_item_graph_dataset import CHECKPOINT


CHECKPOINT = 10


def training_data_masking(ratings: torch.Tensor, df_train: DataFrame):
    trained_positive_items = (
        df_train.groupby("user")["positive"].apply(list).reset_index()
    )
    trained_positive_items["size"] = trained_positive_items["positive"].apply(len)
    exclude_users = []
    exclude_items = []
    # iterate through the trained_positive_items dictionary
    # get the user and items
    # append the user to exclude_users list
    # append the items to exclude_items list
    for _, x in trained_positive_items.iterrows():
        exclude_items.extend([r - 1 for r in x["positive"]])
        exclude_users.extend([x["user"] - 1] * x["size"])
        # exclude_users.extend([user] * len(items))
        # exclude_items.extend(items)

    ratings[exclude_users, exclude_items] = -(1 << 10)
    return ratings


def get_r_at_k(test_positive_items: DataFrame, masked_rating, k=5):
    _rating_values, top_K_items = torch.topk(masked_rating, k)

    r = []
    for _, x in test_positive_items.iterrows():
        ground_truth_items = [v - 1 for v in x["positive"]]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[x["user"] - 1]))
        r.append(label)
        # print(label)
        # print(x["positive"])
        # print(x["user"])
    r = torch.Tensor(np.array(r).astype("float"))
    return r


def hr_at_k(r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    precision = torch.mean(num_correct_pred) / k
    return precision


def recall_at_k(test_positive_items: DataFrame, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor(
        [
            len(test_positive_items.iloc[i]["positive"])
            for i in range(len(test_positive_items))
        ]
    )
    recall = torch.mean(num_correct_pred / user_num_liked)
    return recall.item()


def ndgc_k(items_ground_truth, items_prediction, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        items_ground_truth (list): list of lists containing highly rated items of each user
        items_prediction (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """

    assert len(items_prediction) == len(items_ground_truth)

    test_matrix = torch.zeros((len(items_prediction), k))

    for index, x in items_ground_truth.iterrows():
        length = min(len(x["positive"]), k)
        test_matrix[index, :length] = 1
    max_r = test_matrix

    idcg = torch.sum(max_r * 1.0 / torch.log2(torch.arange(2, k + 2)), axis=1)  # type: ignore
    dcg = items_prediction * (1.0 / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)  # type: ignore
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.0
    return torch.mean(ndcg).item()


def get_metrics(
    model: nn.Module,
    df_test_user_items: DataFrame,
    df_train_user_items: DataFrame,
    top_ks=[5],
):
    user_embedding = model.user_embedding.weight
    item_embedding = model.item_embedding.weight
    ratings = torch.matmul(user_embedding, item_embedding.T)

    return get_metrics_by_rating(
        ratings, df_test_user_items, df_train_user_items, top_ks
    )


def get_metrics_by_rating(
    ratings: torch.Tensor,
    df_test_user_items: DataFrame,
    df_train_user_items: DataFrame,
    top_ks=[5],
):
    rating_masked_train = training_data_masking(ratings, df_train_user_items)

    test_positive_items = (
        df_test_user_items.groupby("user")["positive"].apply(list).reset_index()
    )

    metrics = []

    for top_k in top_ks:
        print("top k", top_k)

        r = get_r_at_k(test_positive_items, rating_masked_train, top_k)
        hr = hr_at_k(r, top_k)
        recall = recall_at_k(test_positive_items, r, top_k)
        ndgc = ndgc_k(test_positive_items, r, top_k)

        print(f"HR@{top_k}: {hr}")
        print(f"Recall@{top_k}: {recall}")
        print(f"NDCG@{top_k}: {ndgc}")
        metrics.append((hr, recall, ndgc))

    return metrics


def save_model(model, save_as_path):
    dirname = os.path.dirname(os.path.abspath(save_as_path))
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), save_as_path)


def model_train_cold_start(
    loader: DataLoader,
    model,
    optimizer,
    epochs=20,
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

            if idx % CHECKPOINT == 0:
                if wandb:
                    wandb.log({"loss": loss})

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
                metrics = get_metrics(model, df_cold, df_train, [5])  # type: ignore
                cold_metrics_plot.append(metrics)

    return loss_plot, test_metrics_plot, cold_metrics_plot
