#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import numpy as np


class Accumulator:
    def __init__(self):
        """Mean and variance accumulator using Welford's algorithm."""
        self.K = None
        self.N = None
        self.ex = None
        self.ex2 = None

    def add(self, datum):
        """ Add a datum to the online accumulator.

        Note: this method can reliably deal with nan numbers, in that it will
        not count these elements as samples.

        :param datum: A datapoint in the form of a list or numpy array.
        """
        # TODO: Algorithms for calculating variance on Wikipedia has more stable
        #     algorithm
        if type(datum) is not np.ndarray:
            datum = np.array(datum)

        isnan = np.isnan(datum)
        if isnan.any() and not isnan.all():
            if self.N is None:
                self.K = np.nan_to_num(datum)
                self.ex = np.zeros(datum.shape)
                self.ex2 = np.zeros(datum.shape)
                self.N = np.zeros(datum.shape)
            else:
                no_nan_datum = np.nan_to_num(datum)
                self.ex += np.isfinite(datum) * (no_nan_datum - self.K)
                self.ex2 += np.isfinite(datum) * ((no_nan_datum - self.K) ** 2)

            self.N += np.isfinite(datum)
        else:
            if self.N is None:
                self.N = np.zeros(datum.shape)
                self.K = datum
                self.ex = np.zeros(datum.shape)
                self.ex2 = np.zeros(datum.shape)
            else:
                self.ex += datum - self.K
                self.ex2 += (datum - self.K) ** 2

            self.N += 1

    # TODO: Add remove method that will work for NaNs, if possible.
    # def remove(self, datum):
    #     """ Add a datum to the online accumulator
    #
    #     :param datum: A datapoint in the form of a list or numpy array.
    #     """
    #     if datum.shape != self.K.shape:
    #         raise ValueError("Array shape does not match.")
    #     self.n -= 1
    #     self.ex -= (datum - self.K)
    #     self.ex2 -= (datum - self.K) * (datum - self.K)

    def get_mean(self):
        if len(np.where(self.N == 0)[0]) > 0:
            print("Some elements have no data points, warnings can be ignored.")

        return self.K + (self.ex / self.N)

    def get_variance(self, exact=True):
        if len(np.where(self.N == 0)[0]) > 0:
            print("Some elements have no data points, warnings can be ignored.")
        if exact:
            return (self.ex2 - (self.ex * self.ex) / self.N) / self.N
        else:
            return (self.ex2 - (self.ex * self.ex) / self.N) / (self.N - 1)


class WelfordAccumulator:
    def __init__(self, min_valid_samples=100):
        """Mean and variance accumulator using Welford's algorithm."""
        self.n = None
        self.mu = None
        self.M2 = None
        self.min_samples = min_valid_samples

    def add(self, datum):
        """ Add a datum to the online accumulator.

        Note: this method can reliably deal with nan numbers, in that it will
        not count these elements as samples.

        :param datum: A datapoint in the form of a list or numpy array.
        """
        #     algorithm
        if type(datum) is not np.ndarray:
            datum = np.array(datum)

        if self.n is None:
            self.n = np.zeros(datum.shape, dtype=np.uint32)
            self.mu = np.zeros(datum.shape, dtype=np.float32)
            self.M2 = np.zeros(datum.shape, dtype=np.float32)

        if np.isnan(datum).any():
            mask = np.isfinite(datum).astype(self.n.dtype)
            self.n += mask
            no_nan_datum = np.nan_to_num(datum)
            delta = no_nan_datum - self.mu
            self.mu += \
                mask * (delta / np.clip(self.n, 1, np.iinfo(self.n.dtype).max))
            delta2 = no_nan_datum - self.mu
            self.M2 += mask * delta * delta2
        else:
            self.n += np.ones(self.n.shape, dtype=self.n.dtype)
            delta = datum - self.mu
            self.mu += delta / self.n
            delta2 = datum - self.mu
            self.M2 += delta * delta2

    def get_mean(self):
        return self.mu * (self.get_valid_mask().astype(np.float32))

    def get_variance(self, exact=True):
        """
        Get the variance for the samples.

        :param exact:
        :return:
        """
        # Find all values with less than minimum elements
        valid = self.get_valid_mask().astype(np.float32)
        invalid = np.logical_not(self.get_valid_mask()).astype(np.float32)

        # Get the solution as if no nans
        var = self.M2 / (np.clip(self.n, 1, np.iinfo(self.n.dtype).max))

        # Set the invalid values to unit variance
        return (valid * var) + invalid

    def get_valid_mask(self):
        return self.n > self.min_samples
