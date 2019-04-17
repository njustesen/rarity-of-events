import glob
import matplotlib.pyplot as plt
import brewer2mpl
import math
import numpy as np
from time import sleep
from event_buffer import EventBufferSQLProxy
from scipy.signal import savgol_filter

fontsize = 14

skip_main_plot = False
skip_event_plot = False
single = False
single_title = "deathmatch"

plt.style.use('ggplot')

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

num_events = 26
num_agents = 2
exp_id = 1000

names = [
    'movement',
    'health',
    'armour',
    'shots',
    'ammo',
    'weapon 0 pickup',
    'weapon 1 pickup',
    'weapon 2 pickup',
    'weapon 3 pickup',
    'weapon 4 pickup',
    'weapon 5 pickup',
    'weapon 6 pickup',
    'weapon 7 pickup',
    'weapon 8 pickup',
    'weapon 9 pickup',
    'kill',
    'weapon 0 kill',
    'weapon 1 kill',
    'weapon 2 kill',
    'weapon 3 kill',
    'weapon 4 kill',
    'weapon 5 kill',
    'weapon 6 kill',
    'weapon 7 kill',
    'weapon 8 kill',
    'weapon 9 kill'
]

data = []
for i in range(num_agents):
    buffer = EventBufferSQLProxy(num_events, 100000000, exp_id, i+1)
    events = buffer.get_own_events()
    data.append(events)

print(data)

axes = []
for i in range(num_events):
    if i >= len(names):
        break
    name = names[i].title()
    plt.title(name)
    cols = math.floor(math.sqrt(num_events))
    rows = math.ceil(math.sqrt(num_events))
    for a in range(num_agents):
        x = []
        y = []
        for e in data[a]:
            x.append(e[0])
            y.append(e[i+1])
        yhat = savgol_filter(y, 5001, 3)  # window size 51, polynomial order 3
        plt.plot(x, yhat, color=colors[a])
    plt.savefig(f'plots/events_{exp_id}_{name}')
    plt.clf()
