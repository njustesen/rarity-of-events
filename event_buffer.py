import numpy as np
#from vizdoom import *
import math
import mysql.connector


class EventBufferSQLProxy:

    def __init__(self, n, capacity, exp_id, actor_id, user='roe', password='RarityOfEvents', host='localhost', database='roe', event_clip=0.01):
        self.n = n
        self.exp_id = exp_id
        self.actor_id = actor_id
        self.capacity = capacity
        self.events = []
        self.event_clip = event_clip
        self.host = host
        self.user = user
        self.password = password
        self.mydb = mysql.connector.connect(
            host=host,
            user=user,
            passwd=password,
            database=database
        )
        self.cache = None
        
    def record_events(self, events, frame):
        mycursor = self.mydb.cursor()
        cmd = "INSERT INTO Event (ExperimentID, ActorID, Frame"
        for i in range(len(events)):
            cmd += ", Event{}".format(i)
        cmd += ")"
        cmd += " VALUES ({}, {}, {}".format(self.exp_id, self.actor_id, frame)
        for i in range(len(events)):
            cmd += ", " + str(events[i])
        cmd += ")"
        mycursor.execute(cmd)
        self.mydb.commit()
        self.cache = None

    def get_events(self):
        mycursor = self.mydb.cursor()
        rows = ""
        for i in range(self.n):
            if i > 0:
                rows += ", "
            rows += "Event{}".format(i)
        cmd = "SELECT " + rows + " FROM Event WHERE ExperimentID = {} AND ActorID != {} ORDER BY EventID DESC LIMIT {}".format(self.exp_id, self.actor_id, self.capacity)
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        events = []
        for x in results:
            if len(events) < self.capacity:
                events.append(x)
            else:
                break
        return events

    def intrinsic_reward(self, events, vector=False):
        if self.cache is None:
            e = self.get_events()
            self.cache = e
        else:
            e = self.cache
        if len(e) == 0:
            if vector:
                return np.ones(self.n)
            return 0
        mean = np.mean(e, axis=0)
        clip = np.clip(mean, self.event_clip, np.max(mean))
        div = np.divide(np.ones(self.n), clip)
        mul = np.multiply(div, events)
        if vector:
            return mul
        return np.sum(mul)

    def get_event_mean(self):
        if self.cache is None:
            e = self.get_events()
            self.cache = e
        else:
            e = self.cache
        if len(e) == 0:
            return np.zeros(self.n)
        mean = np.mean(e, axis=0)
        return mean

    def get_event_rewards(self):
        return self.intrinsic_reward(np.ones(self.n), vector=True)


class EventBuffer:

    def __init__(self, n, capacity, event_clip=0.01):
        self.n = n
        self.capacity = capacity
        self.idx = 0
        self.events = []
        self.event_clip = event_clip

    def record_events(self, events, frame):
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
            return np.zeros(self.n)
        mean = np.mean(self.events, axis=0)
        return mean

    def get_event_rewards(self):
        return self.intrinsic_reward(np.ones(self.n), vector=True)

'''
buffer_0 = EventBufferSQLProxy(2, 100, 11, 0)
buffer_1 = EventBufferSQLProxy(2, 100, 11, 1)
for i in range(20):
    r0 = buffer_0.intrinsic_reward(np.ones(2), vector=True)
    r1 = buffer_1.intrinsic_reward(np.ones(2), vector=True)
    print("R0:", r0)
    print("R1:", r1)
    buffer_0.record_events([1, i*10], frame=i*100)
    buffer_1.record_events([i*10, 1], frame=i*100)
'''