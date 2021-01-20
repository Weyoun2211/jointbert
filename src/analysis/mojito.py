import re

import pandas as pd
import scipy as sp
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict

from lime.lime_text import LimeTextExplainer, IndexedString


class Mojito(LimeTextExplainer):

    def __init__(self,
                 columns,
                 attr_to_copy='left',
                 lprefix='ltable_',
                 rprefix='rtable_',
                 ignore_columns=['id', 'label'],
                 **argv,
                 ):
        super().__init__(**argv)

        self.left = 'L'
        self.right = 'R'
        self.attr_pattern = re.compile(
            '([' + self.left + self.right + '])([0-9]+)\|(.*)$')

        self.lprefix = lprefix
        self.rprefix = rprefix
        self.columns = list(filter(lambda c: c not in ignore_columns, columns))
        self.schema = {}
        self.attr_to_copy = attr_to_copy

        self.__init_schema()

    def copy(self, wrapper_classifier, instances, num_features, num_perturbation, distance_metric='cosine'):
        _wrapper_classifier = lambda strings: wrapper_classifier(self.str_to_pair_of_tuples(strings))
        return self.__explain_group(self.explain_instance_copy,
                                    _wrapper_classifier,
                                    instances,
                                    num_features,
                                    num_perturbation,
                                    distance_metric)

    def drop(self, wrapper_classifier, instances, num_features, num_perturbation, distance_metric='cosine'):
        _wrapper_classifier = lambda strings: wrapper_classifier(self.str_to_pair_of_tuples(strings))
        return self.__explain_group(self.explain_instance_drop,
                                    _wrapper_classifier,
                                    instances,
                                    num_features,
                                    num_perturbation,
                                    distance_metric)

    def __explain_group(self,
                        explain_instance_fn,
                        wrapper_classifier,
                        instances,
                        num_features,
                        num_perturbation,
                        distance_metric):

        result = pd.DataFrame(columns=['exp',
                                       'token',
                                       'attribute',
                                       'tuple',
                                       'weight',
                                       'data_inx'])

        instances_str = self.pair_of_tuples_to_str(instances)
        for i, instance_str in enumerate(instances_str):
            print ("Explaining {}/{}".format(i, len(instances_str)))

            explanation = explain_instance_fn(instance_str,
                                              wrapper_classifier,
                                              num_features=num_features,
                                              num_samples=num_perturbation,
                                              distance_metric=distance_metric)

            for attr_token, weight in explanation.as_list():

                tuple, index, token = self.__split_attr(attr_token)

                result = result.append({
                    'exp':  i,
                    'token': token,
                    'attribute': self.schema[index],
                    'tuple':  tuple,
                    'weight': weight,
                    'data_inx': instances.index[i]
                }, ignore_index=True)

        return result

    def explain_instance_drop(self, *args, **argv):
        def wrapper(*_args, **_argv):
            return LimeTextExplainer._LimeTextExplainer__data_labels_distances(self, *_args, **_argv)

        self._LimeTextExplainer__data_labels_distances = wrapper
        return self.explain_instance(*args, **argv)

    def explain_instance_copy(self, *args, **argv):
        self._LimeTextExplainer__data_labels_distances = self.__data_labels_distances_copy
        return self.explain_instance(*args, **argv)

    def pair_of_tuples_to_str(self, dataframe, ignore_columns=['id', 'label']):
        pairs_strings = []
        for _, row in dataframe.fillna("").iterrows():
            this_pair_str = []
            for tuple_prefix, tuple in [(self.lprefix, 'left'),
                                        (self.rprefix, 'right')]:
                for inx, attr in self.schema.items():

                    tokens = map(lambda token: self.__make_attr(inx, token, tuple=tuple),
                                 row[tuple_prefix + attr].split())

                    this_pair_str += list(tokens)
            pairs_strings.append(" ".join(this_pair_str))

        return pairs_strings

    def str_to_pair_of_tuples(self, strs):
        dataframe = pd.DataFrame(columns = self.columns)
        for row in strs:
            this_pair_row = defaultdict(str)
            row = IndexedString(row, split_expression = " ")
            for attr_index in self.schema.keys():
                left_tokens  = self.__indexes_of_attr(row, self.__make_attr(attr_index, tuple = "left"))
                right_tokens = self.__indexes_of_attr(row, self.__make_attr(attr_index, tuple = "right"))

                this_pair_row[self.lprefix + self.schema[attr_index]] = " ".join([self.__split_attr(row.word(t))[2] for t in left_tokens])
                this_pair_row[self.rprefix + self.schema[attr_index]] = " ".join([self.__split_attr(row.word(t))[2] for t in right_tokens])

            dataframe = dataframe.append(this_pair_row, ignore_index = True)
        return dataframe

    def __data_labels_distances_copy(self,
                                     indexed_string,
                                     classifier_fn,
                                     num_samples,
                                     distance_metric='cosine'):
        def distance_fn(x):
            return pairwise_distances(x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_words()

        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)
        inverse_data = [indexed_string.raw_string()]

        features_range = [self.__make_attr(c, tuple=self.attr_to_copy) for c in self.schema.keys()]

        sample = self.random_state.randint(1, len(features_range) + 1, num_samples - 1)

        for i, size in enumerate(sample, start=1):

            sources = self.random_state.choice(features_range, size, replace=False)
            destinations = list(map(self.__swap_attr, sources))

            # collect all indexes of tokens in destination attributes, then drop them
            tot_tokens_to_drop_inx = []
            for dest_attr in destinations:
                tot_tokens_to_drop_inx += self.__indexes_of_attr(indexed_string, dest_attr)

            data[i, tot_tokens_to_drop_inx] = 0
            sample_without_dest = indexed_string.inverse_removing(tot_tokens_to_drop_inx)

            # collect all indexes of tokens in sources attributes, then copy them (swapping
            # attribute)
            tot_tokens_to_copy_inx = []
            for source_attr in sources:
                tot_tokens_to_copy_inx += self.__indexes_of_attr(
                    indexed_string, source_attr)

            copied_tokens = [self.__swap_attr(
                indexed_string.word(w)) for w in tot_tokens_to_copy_inx]

            # note: since we are using BoW representation, we can safely append anywhere since
            # order do not matter
            sample_for_classifier = " ".join([sample_without_dest, *copied_tokens])
            inverse_data.append(sample_for_classifier)

        # call the classifier & compute distances
        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances

    def __indexes_of_attr(self, indexed_string, attr):
        tokens_indexes = []
        for j in range(indexed_string.num_words()):
            token = indexed_string.word(j)
            if token.startswith(attr):
                tokens_indexes.append(j)
        return tokens_indexes

    def __swap_attr(self, attr):
        if attr.startswith(self.left):
            return self.right + attr[1:]
        return self.left + attr[1:]

    def __make_attr(self, index, value="", tuple='left'):
        if tuple == 'left':
            return self.left + str(index) + "|" + value
        return self.right + str(index) + "|" + value

    def __split_attr(self, value):
        t, index, value = self.attr_pattern.match(value).groups()
        return t, int(index), value

    def __init_schema(self):
        lschema = filter(lambda x: x.startswith(self.lprefix), self.columns)
        for i, s in enumerate(lschema):
            self.schema[i] = s.replace(self.lprefix, "")