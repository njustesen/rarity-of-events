import numpy as np
from vizdoom import *
import math

from arguments import get_args
args = get_args()

class EventBuffer:

    def __init__(self, n, capacity, event_clip=0.01):
        self.n = n
        self.capacity = capacity
        self.idx = 0
        self.events = []
        self.event_clip = event_clip

    def record_events(self, events):
        if len(self.events) < self.capacity:
            self.events.append(events)
        else:
            self.events[self.idx] = events
            if self.idx + 1 < self.capacity:
                self.idx += 1
            else:
                self.idx = 0

    def intrinsic_reward(self, events, vector=False):
        if len(self.events) == 0:
            if vector:
                return np.ones(self.n)
            return 0

        mean = np.mean(self.events, axis=0)
        clip = np.clip(mean, self.event_clip, np.max(mean))
        div = np.divide(np.ones(self.n), clip)
        mul = np.multiply(div, events)
        if vector:
            return mul
        return np.sum(mul)

    def get_event_mean(self):
        if len(self.events) == 0:
            return np.zeros(args.num_events)
        mean = np.mean(self.events, axis=0)
        return mean

    def get_event_rewards(self):
        return self.intrinsic_reward(np.ones(self.n), vector=True)
