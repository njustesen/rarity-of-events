import numpy as np
#from vizdoom import *
import math
import mysql.connector


class Elite:

    def __init__(self, elite_id, fitness):
        self.elite_id = elite_id
        self.fitness = fitness


class EventBufferSQLProxy:

    def __init__(self, n, capacity, exp_id, actor_id, user='roe', password='RarityOfEvents', host='localhost', database='roe', event_clip=0.01, qd=False):
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
        self.qd = qd
        
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
        others = -1
        if self.qd:
            others = self.actor_id
        cmd = "SELECT " + rows + " FROM Archive WHERE ExperimentID = {} AND ActorID != {} ORDER BY EliteID DESC LIMIT {}".format(self.exp_id, others, self.capacity)
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        events = results
        return events
    
    def get_own_events(self):
        mycursor = self.mydb.cursor()
        rows = ""
        for i in range(self.n):
            if i > 0:
                rows += ", "
            rows += "Event{}".format(i)
        #cmd = f"SELECT Frame, unix_timestamp(Timestamp), {rows} FROM Event WHERE ExperimentID = {self.exp_id} AND ActorID = {self.actor_id} ORDER BY EventID ASC"
        cmd = f"SELECT Frame, {rows} FROM Event WHERE ExperimentID = {self.exp_id} AND ActorID = {self.actor_id} ORDER BY EventID ASC"
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        return results

    def add_elite(self, name, events, fitness, frame):
        mycursor = self.mydb.cursor()
        cmd = "INSERT INTO Archive (EliteID, ExperimentID, ActorID, Fitness, Frame"
        for i in range(len(events)):
            cmd += f", Event{i}"
        cmd += ")"
        cmd += f" VALUES ('{name}', {self.exp_id}, {self.actor_id}, {fitness}, {frame}"
        for i in range(len(events)):
            cmd += ", " + str(events[i])
        cmd += ")"
        mycursor.execute(cmd)
        self.mydb.commit()
        self.cache = None

    def get_elite_behaviors(self):
        mycursor = self.mydb.cursor()
        rows = ""
        for i in range(self.n):
            if i > 0:
                rows += ", "
            rows += "Event{}".format(i)
        cmd = f"SELECT {rows} FROM Archive WHERE ExperimentID = {self.exp_id}"
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        return results

    def get_last_own_events_mean(self, n):
        mycursor = self.mydb.cursor()
        rows = ""
        for i in range(self.n):
            if i > 0:
                rows += ", "
            rows += "Event{}".format(i)
        #cmd = f"SELECT Frame, unix_timestamp(Timestamp), {rows} FROM Event WHERE ExperimentID = {self.exp_id} AND ActorID = {self.actor_id} ORDER BY EventID ASC"
        cmd = f"SELECT {rows} FROM Event WHERE ExperimentID = {self.exp_id} AND ActorID = {self.actor_id} ORDER BY EventID DESC LIMIT {n}"
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        return np.mean(results, axis=0)

    def get_max_events(self):
        mycursor = self.mydb.cursor()
        rows = ""
        for i in range(self.n):
            if i > 0:
                rows += ", "
            rows += "MAX(Event{})".format(i)
        #cmd = f"SELECT Frame, unix_timestamp(Timestamp), {rows} FROM Event WHERE ExperimentID = {self.exp_id} AND ActorID = {self.actor_id} ORDER BY EventID ASC"
        cmd = f"SELECT {rows} FROM Archive"
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        if len(results) == 0:
            return []
        return np.mean(results, axis=0)

    def get_neighbors(self, behavior, niche_divs):
        # Get bounds
        mycursor = self.mydb.cursor()
        rows = ""
        for i in range(self.n):
            if i > 0:
                rows += ", "
            rows += "MAX(Event{})".format(i)
        cmd = f"SELECT {rows} FROM Archive WHERE ExperimentID = {self.exp_id}"
        mycursor.execute(cmd)
        maxes = mycursor.fetchall()

        # Get neighbors
        where_rows = ""
        for i in range(self.n):
            min_event = 0
            max_event = maxes[0][i]
            max_event = max(behavior[i], max_event) if max_event is not None else behavior[i]
            cell_size = (max_event - min_event) / niche_divs
            distance = cell_size / 2
            if i > 0:
                where_rows += " AND "
            where_rows += f"Event{i} >= {behavior[i] - distance} AND Event{i} <= {behavior[i] + distance}"
        cmd = f"SELECT Fitness, EliteID FROM Archive WHERE ExperimentID = {self.exp_id} AND {where_rows}"
        mycursor.execute(cmd)
        results = mycursor.fetchall()
        elites = []
        for result in results:
            elites.append(Elite(result[-1], result[-2]))
        return elites

    def remove_elites(self, elites):
        mycursor = self.mydb.cursor()
        rows = "("
        for i in range(len(elites)):
            if i > 0:
                rows += ", "
            rows += f"'{elites[i].elite_id}'"
        rows += ")"
        cmd = f"DELETE FROM Archive where ExperimentID = {self.exp_id} AND EliteID in {rows}"
        mycursor.execute(cmd)

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
