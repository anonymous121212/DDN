import numpy as np
import random as rd


class Data(object):
    def __init__(self, train_file, valid_file, test_file):
        #get number of users and items
        self.n_users, self.n_items = 0, 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    self.n_users += 1
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
            f.close()

        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
            f.close()

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
            f.close()
        self.n_items += 1

        self.train_items, self.valid_set, self.test_set = {}, {}, {}
        with open(train_file) as f_train:
            with open(valid_file) as f_valid:
                with open(test_file) as f_test:
                    for l in f_train.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, train_items = items[0], items[1:]

                        self.train_items[uid] = train_items

                    for l in f_test.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, test_items = items[0], items[1:]
                        self.test_set[uid] = test_items

                    for l in f_valid.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, valid_items = items[0], items[1:]
                        self.valid_set[uid] = valid_items
                    f_train.close()
                    f_valid.close()
                    f_test.close()
        print(self.n_users, self.n_items)

    def sample_pairs(self, batch_size):
        """a sampler to sample a batch size of (users, postive items, negative items)

        Args:
            batch_size (int): batch size

        Returns:
            (users, postive items, negative items)
        """
        if batch_size <= self.n_users:
            users = rd.sample(range(self.n_users), batch_size)
        else:
            users = [rd.choice(range(self.n_users)) for _ in range(batch_size)]


        pos_items = []
        neg_items = []
        for u in users:
            pos_items += self.sample_pos_item_for_u(u)
            neg_items += self.sample_neg_item_for_u(u)

        return users, pos_items, neg_items


    def sample_pos_item_for_u(self, u):
        """a sampler to sample a postive item for user u

        Args:
            u (int): index of user

        Returns:
            a postive item for user u
        """
        pos_items = self.train_items[u]
        return rd.sample(pos_items, 1)

    def sample_neg_item_for_u(self, u):
        """a sampler to sample a negative item for user u

        Args:
            u (int): index of user

        Returns:
            a negative item for user u
        """
        neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
        return rd.sample(neg_items, 1)

    def get_num_users_items(self):
        """a function to return the total number of users and items

        Args:
            None

        Returns:
            number of users, number of items
        """
        return self.n_users, self.n_items