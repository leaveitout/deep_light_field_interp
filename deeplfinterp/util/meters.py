#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import progressbar


class Meter:
    def __init__(self, name: str):
        self.name = name
        self.curr_value = 0.0

    def update(self, next_value: float, n: int = 1):
        self.curr_value = next_value

    def value(self):
        return self.curr_value


class AverageMeter(Meter):
    def __init__(self, name: str, cum: bool = False):
        super().__init__(name)
        self.curr_value = 0.0
        self.sum = 0.0
        self.count = 0
        self.cum = cum

    def update(self, next_value: float, n: int = 1):
        self.curr_value = next_value

        self.sum += self.curr_value * n
        self.count += n

    def value(self):
        if self.cum:
            return self.sum
        else:
            return self.sum / self.count


class AverageAccumMeter(Meter):
    def __init__(self, name):
        """
        Mean accumulator using Welford's algorithm, this meter is somewhat
        safer from overflow errors and higher accuracy when running over a large
        number of samples.
        """
        super().__init__(name)
        self.K = 0.0
        self.N = None
        self.ex = 0.0
        self.curr_value = None

    def update(self, next_value, n=1):
        self.curr_value = next_value

        # TODO: Simplify this and init now we only accept single values.
        if self.N is None:
            self.N = 1
            self.K = self.curr_value
            self.ex = 0.0
        else:
            self.ex += n * (self.curr_value - self.K)
            self.N += n

    def value(self):
        return self.K + (self.ex / self.N)


class CustomProgressBar:
    def __init__(self, label: str = 'Loss'):
        self.format_custom_text = progressbar.FormatCustomText(
            '| ' + label + ': %(value).5f',
            dict(value=0.0),
        )
        self.bar = progressbar.ProgressBar(
            widgets=[progressbar.Percentage(),
                     ' (',
                     progressbar.SimpleProgress(),
                     ') ',
                     progressbar.Bar(),
                     ' ',
                     progressbar.ETA(),
                     ' ',
                     self.format_custom_text
                     ])
