import numpy as np
import tensorflow as tf
import config



def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(np.sum(r))#float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def hr_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r)


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def wasserstein(mean1, cov1, mean2, cov2):
    ret = tf.reduce_sum((mean1 - mean2) * (mean1 - mean2))

    temp = cov1 + cov2 - 2 * tf.sqrt((tf.sqrt(cov1) * cov2 * tf.sqrt(cov1)))
    # temp = tf.sqrt(cov1) - tf.sqrt(cov2)
    ret = ret + tf.reduce_sum(temp)
    return ret

def test_one_user(x):
    # user u's ratings for user u
    ratings = x[0]
    items_to_test = x[1]
    user_pos_test = x[2]

    item_score = [(items_to_test[i], ratings[i]) for i in range(len(items_to_test))]

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    hr_20 = recall_at_k(r, 20, len(user_pos_test))
    ap_20 = average_precision(r, 20)


    return np.array([hr_20,ap_20])
