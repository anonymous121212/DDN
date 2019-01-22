import json
import random
def split(item_list):
    train = item_list[:-2]
    valid = item_list[-2]
    test = item_list[-1]
    return train, [valid], [test]


u2items = {}
item2id = {}
iid = 0
uid = 0
with open('./Video_Games_5.json') as f:
    for line in f.readlines():
        #u, i, r, time = line.split(',')
        data = eval(line)
        u, i, r = data['reviewerID'], data['asin'], data['overall']

        if i not in item2id:
            item2id[i] = iid
            iid += 1

        i = item2id[i]
        if float(r) >= 5.0:
            if u in u2items:
                u2items[u].append(i)
            else:
                u2items[u] = [i]
    f.close()

num_ratings = 0
for u in u2items:
    num_ratings += len(u2items[u])
uid = 0
with open('./train_users.dat', 'w') as f_train:
    with open('./test_users.dat', 'w') as f_test:
        with open('./valid_users.dat', 'w') as f_valid:
            for k, v in u2items.items():
                if len(v) < 3:
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in v]) + '\n')
                else:
                    train, valid, test = split(v)
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in train]) + '\n')
                    f_test.write(str(uid) + ' ' + ' '.join([str(i) for i in test]) + '\n')
                    f_valid.write(str(uid) + ' ' + ' '.join([str(i) for i in valid]) + '\n')
                uid += 1

print('#user %f #item %f density %f' % (uid, iid, float(num_ratings)/(uid*iid)))

uid =0
with open('./train_cold_users.dat', 'w') as f_train:
    with open('./test_cold_users.dat', 'w') as f_test:
        with open('./valid_cold_users.dat', 'w') as f_valid:
            for k, v in u2items.items():
                if len(v) == 1:
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in v]) + '\n')
                elif len(v) == 2:
                    train, test = [v[0]], [v[1]]
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in train]) + '\n')
                    f_test.write(str(uid) + ' ' + ' '.join([str(i) for i in test]) + '\n')
                else:
                    sampled = random.sample(v, 3)
                    train, valid, test = [sampled[0]], [sampled[1]], [sampled[2]]
                    f_train.write(str(uid) + ' ' + ' '.join([str(i) for i in train]) + '\n')
                    f_test.write(str(uid) + ' ' + ' '.join([str(i) for i in test]) + '\n')
                    f_valid.write(str(uid) + ' ' + ' '.join([str(i) for i in valid]) + '\n')
                uid += 1