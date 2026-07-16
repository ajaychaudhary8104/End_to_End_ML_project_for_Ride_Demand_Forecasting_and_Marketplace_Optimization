from collections import Counter

def get_scale_pos_weight(y):

    counter = Counter(y)

    negative = counter[0]

    positive = counter[1]

    return negative / positive