#!/usr/bin/env python3
import json

import numpy as np

from random import sample
from random import shuffle


class Rating():

    def __init__(self, data_tuple):
        self.user, self.item, self.rating, _ = data_tuple
    
    def as_list(self):
        return [self.user, self.item, self.rating]


class RatingCollection():

    SAMPLE_AMOUNT = 0.2
    FOLDS = 10
    FILE = "__tmp__.csv"

    def __init__(self, data_collection):
        self.ratings = [Rating(i) for i in data_collection]
        large_matrix = self._get_matrix(self.ratings)
        self.keys = []
        self.item_keys = None
        with open(self.FILE, 'w') as fp:
            for k, v in large_matrix.items():
                self.keys.append(k)
                if self.item_keys is None:
                    self.item_keys = list(v.keys())
                fp.write("{}\n".format(",".join([str(x) for x in v.values()])))
        del self.ratings

    def _get_matrix(self, ratings):
        user_tuples = {}
        movies = set([r.item for r in self.ratings])
        for rating in ratings:
            if rating.user not in user_tuples.keys():
                user_tuples[rating.user] = {int(r):0 for r in movies}
            user_tuples[rating.user][int(rating.item)] = float(rating.rating) / 5.
        return user_tuples

    def get_sample(self, size):
        indices = sample(list(range(len(self.keys))), size)
        _sample = []
        with open(self.FILE, 'r') as fp:
            for i, line in enumerate(fp):
                if i in indices:
                    _sample.append([float(x) for x in line.split(',')])
        return np.array(_sample)
