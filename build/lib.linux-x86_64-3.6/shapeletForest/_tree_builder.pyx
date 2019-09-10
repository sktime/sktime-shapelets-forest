# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Karlsson

"""
Construct a shapelet based decision tree.
"""

import numpy as np
cimport numpy as np

from libc.math cimport log2
from libc.math cimport INFINITY
from libc.math cimport NAN

from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.string cimport memcpy
from libc.string cimport memset

from shapeletForest._distance cimport TSDatabase
from shapeletForest._distance cimport DistanceMeasure

from shapeletForest._distance cimport ShapeletInfo
from shapeletForest._distance cimport Shapelet

from shapeletForest._distance cimport ts_database_new

from shapeletForest._distance cimport shapelet_info_init
from shapeletForest._distance cimport shapelet_info_free

from shapeletForest._impurity cimport entropy

from shapeletForest._utils cimport label_distribution
from shapeletForest._utils cimport argsort
from shapeletForest._utils cimport rand_int
from shapeletForest._utils cimport RAND_R_MAX


cdef inline SplitPoint new_split_point(size_t split_point,
                                       double threshold,
                                       ShapeletInfo shapelet_info) nogil:
    cdef SplitPoint s
    s.split_point = split_point
    s.threshold = threshold
    s.shapelet_info = shapelet_info
    return s


# pickle a leaf node
cpdef Node remake_classification_leaf_node(size_t n_labels, object proba):
    cdef Node node = Node(NodeType.classification_leaf)
    cdef size_t i
    node.n_labels = n_labels
    node.distribution = <double*> malloc(sizeof(double) * n_labels)
    for i in range(<size_t> proba.shape[0]):
        node.distribution[i] = proba[i]

    return node

cpdef Node remake_regression_leaf_node(double mean_value):
    cdef Node node = Node(NodeType.regression_leaf)
    node.mean_value = mean_value
    return node

# pickle a branch node
cpdef Node remake_branch_node(double threshold, Shapelet shapelet,
                              Node left, Node right):
    cpdef Node node = Node(NodeType.branch)
    node.shapelet = shapelet
    node.threshold = threshold
    node.left = left
    node.right = right
    return node


cdef class Node:
    def __cinit__(self, NodeType node_type):
        self.node_type = node_type
        self.distribution = NULL

    def __dealloc__(self):
        if self.is_classification_leaf and self.distribution != NULL:
            free(self.distribution)
            self.distribution = NULL

    def __reduce__(self):
        if self.is_classification_leaf:
            return (remake_classification_leaf_node,
                    (self.n_labels, self.proba))
        if self.is_regression_leaf:
            return (remake_regression_leaf_node,
                    (self.mean_value,))
        else:
            return (remake_branch_node, (self.threshold,
                                         self.shapelet, self.left, self.right))

    @property
    def is_classification_leaf(self):
        return self.node_type == NodeType.classification_leaf

    @property
    def is_regression_leaf(self):
        return self.node_type == NodeType.regression_leaf

    @property
    def regr_val(self):
        if not self.is_regression_leaf:
            raise AttributeError("not a regression leaf node")
        return self.mean_value

    @property
    def proba(self):
        if not self.is_classification_leaf:
            raise AttributeError("not a classification leaf node")

        cdef np.ndarray[np.float64_t] arr = np.empty(
            self.n_labels, dtype=np.float64)

        cdef size_t i
        for i in range(self.n_labels):
            arr[i] = self.distribution[i]

        return arr


cdef Node new_classification_leaf_node(double* label_buffer,
                                       size_t n_labels,
                                       double n_weighted_samples):
    cdef double* distribution = <double*> malloc(sizeof(double) * n_labels)
    cdef size_t i
    for i in range(n_labels):
        distribution[i] = label_buffer[i] / n_weighted_samples

    cdef Node node = Node(NodeType.classification_leaf)
    node.distribution = distribution
    node.n_labels = n_labels
    return node


cdef Node new_regression_leaf_node(double mean_value):
    cdef Node node = Node(NodeType.regression_leaf)
    node.mean_value = mean_value
    return node


cdef Node new_branch_node(SplitPoint sp, Shapelet shapelet):
    cdef Node node = Node(NodeType.branch)
    node.threshold = sp.threshold
    node.shapelet = shapelet
    return node


cdef class ShapeletTreePredictor:
    cdef size_t n_labels
    cdef DistanceMeasure distance_measure
    cdef TSDatabase td

    def __init__(self,
                 np.ndarray X,
                 DistanceMeasure distance_measure):
        """Construct a shapelet tree predictor

        :param X: the data to predict over
        """
        self.td = ts_database_new(X)
        self.distance_measure = distance_measure

    def predict(self, Node root):
        raise AttributeError("must be overriden")

cdef class ClassificationShapeletTreePredictor(ShapeletTreePredictor):

    def __init__(self,
                 np.ndarray X,
                 DistanceMeasure distance_measure,
                 size_t n_labels):
        super().__init__(X, distance_measure)
        self.n_labels = n_labels

    def predict(self, Node root):
        """Predict the probability of each label using the tree described by
        `root`

        :param root: the root node
        :returns: the probabilities of shape `[n_samples, n_labels]`
        """
        cdef size_t i
        cdef size_t n_samples = self.td.n_samples
        cdef np.ndarray[np.float64_t, ndim=2] output = np.empty(
            [n_samples, self.n_labels], dtype=np.float64)

        cdef Node node
        cdef Shapelet shapelet
        cdef double threshold
        for i in range(n_samples):
            node = root
            while not node.is_classification_leaf:
                shapelet = node.shapelet
                threshold = node.threshold
                if (self.distance_measure.shapelet_distance(
                        shapelet, self.td, i) <= threshold):
                    node = node.left
                else:
                    node = node.right
            output[i, :] = node.proba
        return output

cdef class RegressionShapeletTreePredictor(ShapeletTreePredictor):

    def predict(self, Node root):
        """Predict the probability of each label using the tree described by
        `root`

        :param root: the root node
        :returns: the predictions `[n_samples]`
        """
        cdef size_t i
        cdef size_t n_samples = self.td.n_samples
        cdef np.ndarray[np.float64_t, ndim=1] output = np.empty(
            [n_samples], dtype=np.float64)

        cdef Node node
        cdef Shapelet shapelet
        cdef double threshold
        for i in range(n_samples):
            node = root
            while not node.is_regression_leaf:
                shapelet = node.shapelet
                threshold = node.threshold
                if (self.distance_measure.shapelet_distance(
                        shapelet, self.td, i) <= threshold):
                    node = node.left
                else:
                    node = node.right
            output[i] = node.mean_value
        return output

cdef class ShapeletTreeBuilder:
    cdef size_t random_seed
    cdef size_t n_shapelets
    cdef size_t min_shapelet_size
    cdef size_t max_shapelet_size
    cdef size_t max_depth
    cdef size_t min_sample_split

    cdef size_t label_stride
    cdef size_t n_labels

    cdef double* sample_weights

    cdef TSDatabase td
    cdef size_t n_samples
    cdef size_t* samples
    cdef size_t* samples_buffer
    cdef double n_weighted_samples

    cdef double* distance_buffer

    cdef DistanceMeasure distance_measure

    def __cinit__(self,
                  size_t n_shapelets,
                  size_t min_shapelet_size,
                  size_t max_shapelet_size,
                  size_t max_depth,
                  size_t min_sample_split,
                  DistanceMeasure distance_measure,
                  np.ndarray X,
                  np.ndarray y,
                  np.ndarray sample_weights,
                  object random_state,
                  *args,
                  **kwargs):
        self.random_seed = random_state.randint(0, RAND_R_MAX)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.distance_measure = distance_measure

        self.td = ts_database_new(X)
        self.label_stride = <size_t> y.strides[0] / <size_t> y.itemsize

        self.n_samples = X.shape[0]
        self.samples = <size_t*> malloc(sizeof(size_t) * self.n_samples)
        self.samples_buffer = <size_t*> malloc(
            sizeof(size_t) * self.n_samples)
        self.distance_buffer = <double*> malloc(
            sizeof(double) * self.n_samples)

        if (self.samples == NULL or
            self.distance_buffer == NULL or
            self.samples_buffer == NULL):
            raise MemoryError()

        cdef size_t i
        cdef size_t j = 0
        for i in range(self.n_samples):
            if sample_weights is None or sample_weights[i] != 0.0:
                self.samples[j] = i
                j += 1

        self.n_samples = j
        self.n_weighted_samples = 0

        self.distance_measure = distance_measure
        if sample_weights is None:
            self.sample_weights = NULL
        else:
            self.sample_weights = <double*> sample_weights.data

    def __dealloc__(self):
        free(self.samples)
        free(self.samples_buffer)
        free(self.distance_buffer)

    cpdef Node build_tree(self):
        return self._build_tree(0, self.n_samples, 0)

    cdef Node _make_leaf_node(self, size_t start, size_t end):
        raise ValueError("must be overriden")

    cdef bint _is_pre_pruned(self, size_t start, size_t end):
        raise ValueError("overriden by subclasses")
        
    
    cdef Node _build_tree(self, size_t start, size_t end, size_t depth):
        if self._is_pre_pruned(start, end) or depth >= self.max_depth:
            return self._make_leaf_node(start, end)

        cdef SplitPoint split = self._split(start, end)
        cdef Shapelet shapelet
        cdef Node branch

        cdef double prev_dist
        cdef double curr_dist
        if split.split_point > start and end - split.split_point > 0:
            shapelet = self.distance_measure.get_shapelet(
                split.shapelet_info, self.td)
            branch = new_branch_node(split, shapelet)
            branch.left = self._build_tree(start, split.split_point,
                                           depth + 1)
            branch.right = self._build_tree(split.split_point, end,
                                            depth + 1)

            shapelet_info_free(split.shapelet_info)
            return branch
        else:
            # new_classification_leaf_node(
            #     self.label_buffer, self.n_labels, self.n_weighted_samples)
            return self._make_leaf_node(start, end)

    cdef SplitPoint _split(self, size_t start, size_t end) nogil:
        cdef size_t split_point, best_split_point
        cdef double threshold, best_threshold
        cdef double impurity
        cdef double best_impurity
        cdef ShapeletInfo shapelet
        cdef ShapeletInfo best_shapelet
        cdef size_t i

        shapelet_info_init(&best_shapelet)
        best_impurity = INFINITY
        best_threshold = NAN
        best_split_point = 0
        split_point = 0

        for i in range(self.n_shapelets):
            shapelet = self._sample_shapelet(start, end)
            self.distance_measure.shapelet_info_distances(
                shapelet, self.td, self.samples + start,
                self.distance_buffer + start, end - start)

            argsort(self.distance_buffer + start, self.samples + start,
                    end - start)
            self._partition_distance_buffer(
                start, end, &split_point, &threshold, &impurity)
            if impurity < best_impurity:
                # store the order of samples in `sample_buffer`
                memcpy(self.samples_buffer,
                       self.samples + start, sizeof(size_t) * (end - start))
                best_impurity = impurity
                best_split_point = split_point
                best_threshold = threshold
                best_shapelet = shapelet
            else:
                shapelet_info_free(shapelet)

        # restore the best order to `samples`
        memcpy(self.samples + start,
               self.samples_buffer, sizeof(size_t) * (end - start))
        return new_split_point(best_split_point, best_threshold,
                               best_shapelet)

    cdef ShapeletInfo _sample_shapelet(self, size_t start, size_t end) nogil:
        cdef size_t shapelet_length
        cdef size_t shapelet_start
        cdef size_t shapelet_index
        cdef size_t shapelet_dim

        shapelet_length = rand_int(self.min_shapelet_size,
                                   self.max_shapelet_size,
                                   &self.random_seed)
        shapelet_start = rand_int(0, self.td.n_timestep - shapelet_length,
                                  &self.random_seed)
        shapelet_index = self.samples[rand_int(start, end, &self.random_seed)]
        if self.td.n_dims > 1:
            shapelet_dim = rand_int(0, self.td.n_dims, &self.random_seed)
        else:
            shapelet_dim = 1

        return self.distance_measure.new_shapelet_info(self.td,
                                                       shapelet_index,
                                                       shapelet_start,
                                                       shapelet_length,
                                                       shapelet_dim)

    cdef void _partition_distance_buffer(self,
                                         size_t start,
                                         size_t end,
                                         size_t* split_point,
                                         double* threshold,
                                         double* impurity) nogil:
        # Overriden by subclasses
        return;



cdef class ClassificationShapeletTreeBuilder(ShapeletTreeBuilder):
    
    cdef double* label_buffer
    cdef double* left_label_buffer
    cdef double* right_label_buffer
    cdef size_t* labels

    def __cinit__(self,
                  size_t n_shapelets,
                  size_t min_shapelet_size,
                  size_t max_shapelet_size,
                  size_t max_depth,
                  size_t min_sample_split,
                  DistanceMeasure distance_measure,
                  np.ndarray X,
                  np.ndarray y,
                  np.ndarray sample_weights,
                  object random_state,
                  size_t n_labels):
        self.labels = <size_t*> y.data
        self.n_labels = n_labels
        self.label_buffer = <double*> malloc(sizeof(double) * n_labels)
        self.left_label_buffer = <double*> malloc(sizeof(double) * n_labels)
        self.right_label_buffer= <double*> malloc(sizeof(double) * n_labels)
        if (self.left_label_buffer == NULL or
            self.right_label_buffer == NULL or
            self.label_buffer == NULL):
            raise MemoryError()

    def __dealloc__(self):
        free(self.label_buffer)
        free(self.left_label_buffer)
        free(self.right_label_buffer)

    cdef Node _make_leaf_node(self, size_t start, size_t end):
        return new_classification_leaf_node(
            self.label_buffer, self.n_labels, self.n_weighted_samples)

    cdef bint _is_pre_pruned(self, size_t start, size_t end):
        # reset the `label_buffer`
        memset(self.label_buffer, 0, sizeof(double) * self.n_labels)
        cdef int n_positive = label_distribution(
            self.samples,
            self.sample_weights,
            start,
            end,
            self.labels,
            self.label_stride,
            self.n_labels,
            &self.n_weighted_samples,  # out param
            self.label_buffer,  # out param
        )

        if end - start <= self.min_sample_split or n_positive < 2:
            return True
        return False

    cdef void _partition_distance_buffer(self,
                                         size_t start,
                                         size_t end,
                                         size_t* split_point,
                                         double* threshold,
                                         double* impurity) nogil:
        memset(self.left_label_buffer, 0, sizeof(double) * self.n_labels)

        # store the label buffer temporarily in `right_label_buffer`
        memcpy(self.right_label_buffer, self.label_buffer,
               sizeof(double) * self.n_labels)

        cdef size_t i # real index of samples
        cdef size_t j # sample index
        cdef size_t p # label index

        cdef double right_sum
        cdef double left_sum

        cdef double prev_distance
        cdef size_t prev_label

        cdef double current_sample_weight
        cdef double current_distance
        cdef double current_impurity
        cdef size_t current_label

        j = self.samples[start]
        p = j * self.label_stride

        prev_distance = self.distance_buffer[start]
        prev_label = self.labels[p]

        if self.sample_weights != NULL:
            current_sample_weight = self.sample_weights[j]
        else:
            current_sample_weight = 1.0

        left_sum = current_sample_weight
        right_sum = self.n_weighted_samples - current_sample_weight

        self.left_label_buffer[prev_label] += current_sample_weight
        self.right_label_buffer[prev_label] -= current_sample_weight

        impurity[0] = entropy(left_sum,
                              self.left_label_buffer,
                              right_sum,
                              self.right_label_buffer,
                              self.n_labels)

        threshold[0] = prev_distance # / 2 removed - this changes the trees
        split_point[0] = start + 1 # The split point indicates a <=-relation

        for i in range(start + 1, end - 1):
            j = self.samples[i]
            current_distance = self.distance_buffer[i]

            p = j * self.label_stride
            current_label = self.labels[p]

            if not current_label == prev_label:
                current_impurity = entropy(left_sum,
                                           self.left_label_buffer,
                                           right_sum,
                                           self.right_label_buffer,
                                           self.n_labels)

                if current_impurity <= impurity[0]:
                    impurity[0] = current_impurity
                    threshold[0] = (current_distance + prev_distance) / 2
                    split_point[0] = i

            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0

            left_sum += current_sample_weight
            right_sum -= current_sample_weight
            self.left_label_buffer[current_label] += current_sample_weight
            self.right_label_buffer[current_label] -= current_sample_weight

            prev_label = current_label
            prev_distance = current_distance

cdef class RegressionShapeletTreeBuilder(ShapeletTreeBuilder):

    cdef double* labels

    def __cinit__(self,
                  size_t n_shapelets,
                  size_t min_shapelet_size,
                  size_t max_shapelet_size,
                  size_t max_depth,
                  size_t min_sample_split,
                  DistanceMeasure distance_measure,
                  np.ndarray X,
                  np.ndarray y,
                  np.ndarray sample_weights,
                  object random_state):
        self.labels = <double*> y.data

    cdef bint _is_pre_pruned(self, size_t start, size_t end):
        cdef size_t j
        cdef double sample_weight
        self.n_weighted_samples = 0
        for i in range(start, end):
            j = self.samples[i]
            if self.sample_weights != NULL:
                sample_weight = self.sample_weights[j]
            else:
                sample_weight = 1.0
                
            self.n_weighted_samples += sample_weight
        if end - start < self.min_sample_split:
            return True
        return False

    cdef Node _make_leaf_node(self, size_t start, size_t end):
        cdef double leaf_sum = 0
        cdef double current_sample_weight = 0
        cdef size_t j
        for i in range(start, end):
            j = self.samples[i]
            p = j * self.label_stride
            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0
            leaf_sum += self.labels[p] * current_sample_weight

        return new_regression_leaf_node(leaf_sum / self.n_weighted_samples)

    cdef void _partition_distance_buffer(self,
                                         size_t start,
                                         size_t end,
                                         size_t* split_point,
                                         double* threshold,
                                         double* impurity) nogil:
        cdef size_t i # real index of samples
        cdef size_t j # sample index
        cdef size_t p # label index

        cdef double right_sum
        cdef double right_mean
        cdef double right_mean_old
        cdef double right_var
        cdef double left_sum
        cdef double left_mean
        cdef double left_mean_old
        cdef double left_var

        cdef double prev_distance

        cdef double current_sample_weight
        cdef double current_distance
        cdef double current_impurity
        cdef double current_val

        j = self.samples[start]
        p = j * self.label_stride

        prev_distance = self.distance_buffer[start]
        current_val = self.labels[p]

        if self.sample_weights != NULL:
            current_sample_weight = self.sample_weights[j]
        else:
            current_sample_weight = 1.0

        left_sum = current_sample_weight
        left_mean_old = 0
        left_mean = current_val
        left_var = 0 
        
        right_sum = 0
        right_mean_old = 0
        right_mean = 0
        right_var = 0
        
        for i in range(start + 1, end):
            j = self.samples[i]
            p = j * self.label_stride
            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0
            
            current_val = self.labels[p]
            right_sum += current_sample_weight
            right_mean_old = right_mean
            right_mean = (right_mean_old +
                          (current_sample_weight / right_sum) *
                          (current_val - right_mean_old))
            right_var += (current_sample_weight *
                          (current_val - right_mean_old) *
                          (current_val - right_mean))

        impurity[0] = (left_var / left_sum) + (right_var / right_sum)
        threshold[0] = prev_distance
        split_point[0] = start + 1 # The split point indicates a <=-relation

        for i in range(start + 1, end - 1):
            j = self.samples[i]
            p = j * self.label_stride
            
            current_distance = self.distance_buffer[i]

            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0


            current_val = self.labels[p]
            right_sum -= current_sample_weight
            right_mean_old = right_mean
            right_mean = (right_mean_old -
                          (current_sample_weight / right_sum) *
                          (current_val - right_mean_old))
            right_var -= (current_sample_weight *
                          (current_val - right_mean_old) *
                          (current_val - right_mean))

            left_sum += current_sample_weight
            left_mean_old = left_mean
            left_mean = (left_mean_old +
                         (current_sample_weight / left_sum) *
                         (current_val - left_mean_old))
            left_var += (current_sample_weight *
                         (current_val - left_mean_old) *
                         (current_val - left_mean))

            prev_distance = current_distance 
            current_impurity = (left_var / left_sum) + (right_var / right_sum)

            if current_impurity <= impurity[0]:
                impurity[0] = current_impurity
                threshold[0] = (current_distance + prev_distance) / 2
                split_point[0] = i

