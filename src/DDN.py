import numpy as np
import tensorflow as tf
import utils as ut
import random as rd
import config
import multiprocessing
cores = multiprocessing.cpu_count()

class DDN(object):
    def __init__(self, data, batch_size, n_layers):
        self.data = data
        self.n_users, self.n_items = self.data.get_num_users_items()
        self.batch_size = batch_size
        self.n_layers = n_layers

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.neg_items = tf.placeholder(tf.int32, shape=(self.batch_size,))

        # variable definition
        self.user_feats = tf.Variable(
            tf.random_normal([self.n_users, config.shape[0]], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_feats')
        self.item_feats = tf.Variable(
            tf.random_normal([self.n_items, config.shape[0]], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_feats')

        self.mean_weights_user = [tf.Variable(
            tf.random_normal([config.shape[l], config.shape[l+1]], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(n_layers-1)]

        self.mean_bias_user = [tf.Variable(
            tf.random_normal([config.shape[l],], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(1, n_layers)]

        self.cov_weights_user = [tf.Variable(
            tf.random_normal([config.shape[l], config.shape[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(n_layers - 1)]

        self.cov_bias_user = [tf.Variable(
            tf.random_normal([config.shape[l], ], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(1, n_layers)]

        self.mean_weights_item = [tf.Variable(
            tf.random_normal([config.shape[l], config.shape[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(n_layers - 1)]

        self.mean_bias_item = [tf.Variable(
            tf.random_normal([config.shape[l], ], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(1, n_layers)]

        self.cov_weights_item = [tf.Variable(
            tf.random_normal([config.shape[l], config.shape[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(n_layers - 1)]

        self.cov_bias_item = [tf.Variable(
            tf.random_normal([config.shape[l], ], mean=0.01, stddev=0.02, dtype=tf.float32))
            for l in range(1, n_layers)]

        u_feats = tf.gather(self.user_feats, self.users)
        pos_feats = tf.gather(self.item_feats, self.pos_items)
        neg_feats = tf.gather(self.item_feats, self.neg_items)
        self.u_mean = self.mean_network_user(embeddings=u_feats)
        self.u_cov = self.cov_network_user(embeddings=u_feats)

        self.pos_items_mean = self.mean_network_item(embeddings=pos_feats)
        self.pos_items_cov = self.cov_network_item(embeddings=pos_feats)

        self.neg_items_mean = self.mean_network_item(embeddings=neg_feats)
        self.neg_items_cov = self.cov_network_item(embeddings=neg_feats)

        # define loss
        losses = []
        distance = []
        for b in range(self.batch_size):
            u_mean, u_cov = tf.gather(self.u_mean, b), tf.gather(self.u_cov, b)
            pos_item_mean, pos_item_cov = tf.gather(self.pos_items_mean, b), tf.gather(self.pos_items_cov, b)
            neg_item_mean, neg_item_cov = tf.gather(self.neg_items_mean, b), tf.gather(self.neg_items_cov, b)
            self.dis1 = ut.wasserstein(mean1=u_mean, cov1=u_cov, mean2=pos_item_mean, cov2=pos_item_cov)
            distance.append(self.dis1)
            self.dis2 = ut.wasserstein(mean1=u_mean, cov1=u_cov, mean2=neg_item_mean, cov2=neg_item_cov)
            loss = -tf.log(tf.nn.sigmoid(self.dis2 - self.dis1))
            losses.append(loss)
        self.distance = tf.stack(distance)
        self.loss = tf.reduce_sum(losses) + config.lamda * (
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.mean_weights_user]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.mean_bias_user]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.cov_weights_user]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.cov_bias_user]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.mean_weights_item]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.mean_bias_item]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.cov_weights_item]) +
                    tf.reduce_sum([tf.nn.l2_loss(w) for w in self.cov_bias_item])
        )


    def train(self, n_epoch, lr, optimizer):
        """function for training

        Args:
            n_epoch (int): number of epoch
            lr (float): learning rate
            optimizer: prefered optimization methods

        Returns:
            None
        """
        assert optimizer in {'Adam', 'SGD'}

        if optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(lr)
        if optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(lr)

        self.updates = self.opt.minimize(self.loss)

        configuration = tf.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configuration)
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            users, pos_items, neg_items = self.data.sample_pairs(self.batch_size)
            _, loss, u_cov = self.sess.run([self.updates, self.loss, self.u_cov],
                                   feed_dict={self.users: users,
                                              self.pos_items: pos_items,
                                              self.neg_items: neg_items})
            print("Epoch %d loss %f" % (epoch, loss))


    def mean_network_user(self, embeddings):
        """function for building a mean network for users

        Args:
            embeddings (tensorflow variable): user features

        Returns:
            a mean vector of users
        """

        mean = embeddings
        for l in range(self.n_layers-1):
            weights = self.mean_weights_user[l]
            bias = self.mean_bias_user[l]
            mean = tf.nn.elu(tf.matmul(mean, weights) + bias)
        return mean


    def cov_network_user(self, embeddings):
        """function for building a covariance network for users

        Args:
            embeddings (tensorflow variable): user features

        Returns:
            a diagonal covariance of users
        """
        cov = embeddings
        for l in range(self.n_layers-1):
            weights = self.cov_weights_user[l]
            bias = self.cov_bias_user[l]
            cov = tf.nn.elu(tf.matmul(cov, weights) + bias) + 1
        return cov


    def mean_network_item(self, embeddings):
        """function for building a mean network for items

        Args:
            embeddings (tensorflow variable): item features

        Returns:
            a mean vector of items
        """
        mean = embeddings
        for l in range(self.n_layers-1):
            weights = self.mean_weights_item[l]
            bias = self.mean_bias_item[l]
            mean = tf.nn.elu(tf.matmul(mean, weights) + bias)
        return mean


    def cov_network_item(self, embeddings):
        """function for building a covariance network for items

        Args:
            embeddings (tensorflow variable): item features

        Returns:
            a diagonal covariance of items
        """
        cov = embeddings
        for l in range(self.n_layers-1):
            weights = self.cov_weights_item[l]
            bias = self.cov_bias_item[l]
            cov = tf.nn.elu(tf.matmul(cov, weights) + bias) + 1
        return cov


    def predict(self,mode):
        """function for prediction based mode

        Args:
            mode (str): validation when mode is set to "valid", otherwise testing

        Returns:
            None
        """

        result = np.array([0.] * 15)
        pool = multiprocessing.Pool(cores)
        # all users needed to test
        if mode == 'valid':
            test_users = list(self.data.valid_set.keys())
        else:
            test_users = list(self.data.test_set.keys())

        test_user_num = len(test_users)
        for u in test_users:
            users = [u] * self.batch_size
            user_pos_test = self.data.test_set[u] if mode == 'test' else self.data.valid_set[u]

            neg_items = set(range(self.data.n_items)) - set(self.data.train_items[u])
            if u in self.data.valid_set: neg_items = neg_items - set(self.data.valid_set[u])
            if u in self.data.test_set: neg_items = neg_items - set(self.data.test_set[u])
            neg_items = rd.sample(neg_items, self.batch_size-1)
            items_to_test = neg_items + user_pos_test
            ratings = self.sess.run(self.distance, {self.users: users, self.pos_items: items_to_test})

            item_score = [(items_to_test[i], ratings[i]) for i in range(len(items_to_test))]

            item_score = sorted(item_score, key=lambda x: x[1])
            #item_score.reverse()
            item_sort = [x[0] for x in item_score]

            r = []
            for i in item_sort:
                if i in user_pos_test:
                    r.append(1)
                else:
                    r.append(0)

            hr_1 = ut.hr_at_k(r, 1)
            hr_3 = ut.hr_at_k(r, 3)
            hr_5 = ut.hr_at_k(r, 5)
            hr_7 = ut.hr_at_k(r, 7)
            hr_10 = ut.hr_at_k(r, 10)

            ap_1 = ut.average_precision(r, 1)
            ap_3 = ut.average_precision(r, 3)
            ap_5 = ut.average_precision(r, 5)
            ap_7 = ut.average_precision(r, 7)
            ap_10 = ut.average_precision(r, 10)

            ndcg_1 =  ut.ndcg_at_k(r, 1)
            ndcg_3 = ut.ndcg_at_k(r, 3)
            ndcg_5 = ut.ndcg_at_k(r, 5)
            ndcg_7 = ut.ndcg_at_k(r, 7)
            ndcg_10 = ut.ndcg_at_k(r, 10)
            ret = np.array([hr_1, hr_3, hr_5, hr_7, hr_10,
                            ap_1, ap_3, ap_5, ap_7, ap_10,
                            ndcg_1, ndcg_3, ndcg_5, ndcg_7, ndcg_10])
            result += ret


        ret = result / test_user_num

        hr_1, hr_3, hr_5, hr_7, hr_10 = ret[0], ret[1], ret[2], ret[3], ret[4]
        map_1, map_3, map_5, map_7, map_10 = ret[5], ret[6], ret[7], ret[8], ret[9]
        ndcg_1, ndcg_3, ndcg_5, ndcg_7, ndcg_10 = ret[10], ret[11], ret[12], ret[13], ret[14]

        print('Test:') if mode == 'test' else print('Valid:')
        print('hr@1 %f hr@3 %f hr@5 %f hr@7 %f hr@10 %f' % (hr_1, hr_3, hr_5, hr_7, hr_10))
        print('MAP@1 %f MAP@3 %f MAP@5 %f MAP@7 %f MAP@10 %f' % (map_1, map_3, map_5, map_7, map_10))
        print('ndcg@1 %f ndcg@3 %f ndcg@5 %f ndcg@7 %f ndcg@10 %f' % (ndcg_1, ndcg_3, ndcg_5, ndcg_7, ndcg_10))

