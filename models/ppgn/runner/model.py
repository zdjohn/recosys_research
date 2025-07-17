import tensorflow as tf
import numpy as np
import math


class PPGN(tf.keras.Model): # type: ignore
    def __init__(
        self, args, norm_adj_mat, num_users, num_items_s, num_items_t, **kwargs
    ):
        super(PPGN, self).__init__(**kwargs)
        self.args = args
        self.norm_adj_mat = norm_adj_mat
        self.num_users = num_users
        self.num_items_s = num_items_s
        self.num_items_t = num_items_t
        self.n_fold = 100

        self.all_weights = self.init_weights()
        self.item_embeddings_s, self.user_embeddings, self.item_embeddings_t = (
            self.creat_gcn_embedd()
        )

        # Initialize optimizer
        self.optimizer_obj = tf.keras.optimizers.Adam(learning_rate=self.args.lr)

        # Initialize dense layers for NCF
        self.dense_layers_s = []
        self.dense_layers_t = []
        self.dropout_layers_s = []
        self.dropout_layers_t = []

        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        regularizer = tf.keras.regularizers.l2(self.args.regularizer_rate)

        for i, units in enumerate(self.args.mlp_layers):
            self.dense_layers_s.append(
                tf.keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    name="dense_s_%d" % i,
                )
            )
            self.dropout_layers_s.append(
                tf.keras.layers.Dropout(rate=self.args.dropout_message)
            )

            self.dense_layers_t.append(
                tf.keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    name="dense_t_%d" % i,
                )
            )
            self.dropout_layers_t.append(
                tf.keras.layers.Dropout(rate=self.args.dropout_message)
            )

        self.logits_dense_s_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="logits_dense_s",
        )
        self.logits_dense_t_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="logits_dense_t",
        )

    def init_weights(self):
        all_weights = dict()
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        regularizer = tf.keras.regularizers.l2(self.args.regularizer_rate)

        all_weights["user_embeddings"] = tf.Variable(
            initializer((self.num_users, self.args.embedding_size)),
            name="user_embeddings",
            trainable=True,
        )
        all_weights["item_embeddings_s"] = tf.Variable(
            initializer((self.num_items_s, self.args.embedding_size)),
            name="item_embeddings_s",
            trainable=True,
        )
        all_weights["item_embeddings_t"] = tf.Variable(
            initializer((self.num_items_t, self.args.embedding_size)),
            name="item_embeddings_t",
            trainable=True,
        )

        self.layers_plus = [self.args.embedding_size] + self.args.gnn_layers

        for k in range(len(self.layers_plus) - 1):
            all_weights["W_gc_%d" % k] = tf.Variable(
                initializer((self.layers_plus[k], self.layers_plus[k + 1])),
                name="W_gc_%d" % k,
                trainable=True,
            )
            all_weights["b_gc_%d" % k] = tf.Variable(
                tf.zeros(self.layers_plus[k + 1]), name="b_gc_%d" % k, trainable=True
            )
            all_weights["W_bi_%d" % k] = tf.Variable(
                initializer((self.layers_plus[k], self.layers_plus[k + 1])),
                name="W_bi_%d" % k,
                trainable=True,
            )
            all_weights["b_bi_%d" % k] = tf.Variable(
                tf.zeros(self.layers_plus[k + 1]), name="b_bi_%d" % k, trainable=True
            )

        return all_weights

    def creat_gcn_embedd(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat)
        embeddings = tf.concat(
            [
                self.all_weights["item_embeddings_s"],
                self.all_weights["user_embeddings"],
                self.all_weights["item_embeddings_t"],
            ],
            axis=0,
        )
        all_embeddings = [embeddings]

        for k in range(len(self.layers_plus) - 1):
            temp_embedd = [
                tf.sparse.sparse_dense_matmul(A_fold_hat[f], embeddings)
                for f in range(self.n_fold)
            ]

            embeddings = tf.concat(temp_embedd, axis=0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.all_weights["W_gc_%d" % k])
                + self.all_weights["b_gc_%d" % k]
            )
            embeddings = tf.nn.dropout(embeddings, rate=self.args.dropout_message)

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, axis=1)
        item_embeddings_s, user_embeddings, item_embeddings_t = tf.split(
            all_embeddings, [self.num_items_s, self.num_users, self.num_items_t], axis=0
        )

        return item_embeddings_s, user_embeddings, item_embeddings_t

    def _split_A_hat(self, X):
        fold_len = math.ceil((X.shape[0]) / self.n_fold)
        A_fold_hat = [
            self._convert_sp_mat_to_sp_tensor(
                X[i_fold * fold_len : (i_fold + 1) * fold_len]
            )
            for i_fold in range(self.n_fold)
        ]

        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.array([coo.row, coo.col]).transpose()  # np.mat is deprecated

        return tf.SparseTensor(indices, coo.data, coo.shape)

    def call(self, inputs, training=None):
        """Forward pass of the model

        Args:
            inputs: Dictionary with keys 'user', 'item_s', 'item_t', 'label_s', 'label_t'
            training: Boolean indicating training mode
        """
        user = inputs["user"]
        item_s = inputs["item_s"]
        item_t = inputs["item_t"]
        label_s = tf.cast(inputs["label_s"], tf.float32)
        label_t = tf.cast(inputs["label_t"], tf.float32)

        user_embedding = tf.nn.embedding_lookup(self.user_embeddings, user)
        item_embedding_s = tf.nn.embedding_lookup(self.item_embeddings_s, item_s)
        item_embedding_t = tf.nn.embedding_lookup(self.item_embeddings_t, item_t)

        if self.args.NCForMF == "MF":
            logits_dense_s = tf.reduce_sum(
                tf.multiply(user_embedding, item_embedding_s), 1
            )
            logits_dense_t = tf.reduce_sum(
                tf.multiply(user_embedding, item_embedding_t), 1
            )
        elif self.args.NCForMF == "NCF":
            a_s = tf.concat([user_embedding, item_embedding_s], axis=-1)
            a_t = tf.concat([user_embedding, item_embedding_t], axis=-1)

            for i in range(len(self.args.mlp_layers)):
                a_s = self.dense_layers_s[i](a_s)
                a_s = self.dropout_layers_s[i](a_s, training=training)

                a_t = self.dense_layers_t[i](a_t)
                a_t = self.dropout_layers_t[i](a_t, training=training)

            logits_dense_s = self.logits_dense_s_layer(a_s)
            logits_dense_t = self.logits_dense_t_layer(a_t)
        else:
            raise ValueError

        logits_s = tf.squeeze(logits_dense_s)
        logits_t = tf.squeeze(logits_dense_t)

        return {
            "logits_s": logits_s,
            "logits_t": logits_t,
            "label_s": label_s,
            "label_t": label_t,
        }

    def compute_loss(self, outputs):
        """Compute the loss for training"""
        logits_s = outputs["logits_s"]
        logits_t = outputs["logits_t"]
        label_s = outputs["label_s"]
        label_t = outputs["label_t"]

        loss_list_s = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_s, logits=logits_s
        )
        loss_list_t = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_t, logits=logits_t
        )

        loss_w_s = tf.where(tf.equal(label_s, 1.0), 5.0, 1.0)
        loss_w_t = tf.where(tf.equal(label_t, 1.0), 5.0, 1.0)

        loss_s = tf.reduce_mean(tf.multiply(loss_list_s, loss_w_s))
        loss_t = tf.reduce_mean(tf.multiply(loss_list_t, loss_w_t))

        total_loss = loss_s + loss_t

        # Add regularization losses
        total_loss += sum(self.losses)

        return total_loss

    def get_predictions(self, outputs):
        """Get top-k predictions for evaluation"""
        logits_s = outputs["logits_s"]
        logits_t = outputs["logits_t"]
        label_s = outputs["label_s"]
        label_t = outputs["label_t"]

        _, indice_s = tf.nn.top_k(tf.sigmoid(logits_s), self.args.topK)
        _, indice_t = tf.nn.top_k(tf.sigmoid(logits_t), self.args.topK)

        prediction_s = tf.gather(label_s, indice_s)
        prediction_t = tf.gather(label_t, indice_t)

        return prediction_s, label_s, prediction_t, label_t
