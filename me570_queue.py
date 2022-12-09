"""
A pedagogical implementation of a priority queue
"""

from numbers import Number


class PriorityQueue:
    """ Implements a priority queue """
    def __init__(self):
        """
        Initializes the internal attribute  queue to be an empty list.
        """
        self.queue_list = []

    def check(self):
        """
        Check that the internal representation is a list of (key,value) pairs,
        where value is numerical
        """
        is_valid = True
        for pair in self.queue_list:
            if len(pair) != 2:
                is_valid = False
                break
            if not isinstance(pair[1], Number):
                is_valid = False
                break
        return is_valid

    def insert(self, key, cost):
        """
        Add an element to the queue.
        """
        self.queue_list.append((key, cost))

    def min_extract(self):
        """
        Extract the element with minimum cost from the queue.
        """
        if len(self.queue_list) == 0:
            cost_best = None
            key_best = None
        else:
            idx_best = 0
            cost_best = self.queue_list[0][1]

            for idx in range(1, len(self.queue_list)):
                cost = self.queue_list[idx][1]
                if cost < cost_best:
                    cost_best = cost
                    idx_best = idx

            key_best = self.queue_list[idx_best][0]
            del self.queue_list[idx_best]
        return key_best, cost_best

    def is_member(self, key):
        """
        Check whether an element with a given key is in the queue or not.
        """
        return key in [key for (key, _) in self.queue_list]
