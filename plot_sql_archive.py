import glob
import matplotlib.pyplot as plt
import brewer2mpl
import math
import numpy as np
from time import sleep
from event_buffer import EventBufferSQLProxy
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import imageio
import os
import matplotlib.lines as lines

fontsize = 14

skip_main_plot = False
skip_event_plot = False
single = False
single_title = "deathmatch"

plt.style.use('ggplot')

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

'''
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
'''

names = [
    'movement',
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

num_events = len(names)
num_agents = 2
exp_id = 2
pca = False


buffer = EventBufferSQLProxy(num_events, 100000000, exp_id, 0)
elites = buffer.get_elites()

behaviors = []
fitnesses = []
for i in range(num_agents):
    b = [elite.events for elite in elites if elite.actor == i+1]
    f = [elite.fitness for elite in elites if elite.actor == i+1]
    behaviors.append(b)
    fitnesses.append(f)
    print(f"Agent {i} elites: {len(f)}")
    idx = np.argmax(f)
    agent_elite = [elite for elite in elites if elite.actor == i+1][idx]
    print(f"Best agent {i} has ID {agent_elite.elite_id} and fitness {agent_elite.fitness}")


def plot_pca(data, fitnesses, max_fit=1):
    fig, plot = plt.subplots()
    fig.set_size_inches(4, 4)
    plt.prism()
    for i in range(len(data)):
        for p in range(len(data[i])):
            size = 2 + (fitnesses[i][p] / max_fit) * 8
            plt.plot(data[i][p][0], data[i][p][1], 'o', markerfacecolor=colors[i], markersize=size, fillstyle='full', markeredgewidth=0.0)
    plot.set_xticks(())
    plot.set_yticks(())
    plt.title("Archive")
    plt.tight_layout(pad=-0.5, w_pad=-0.5, h_pad=-0.5)
    #fig.savefig("plots/{}.pdf".format("pca" if pca else "t-sne"), bbox_inches='tight', pad_inches=0)
    fig.savefig("plots/archives/archive.png", bbox_inches='tight', pad_inches=0)
    return fig

# Standardize
y_all = []
max_fit = 0
for a in range(num_agents):
    for i in range(len(behaviors[a])):
        events = behaviors[a][i]
        y_all.append(events)
        if fitnesses[a][i] > max_fit:
            max_fit = fitnesses[a][i]
y_all = StandardScaler().fit_transform(y_all)

# Reduce dimensions
if pca:
    transformed = PCA(n_components=2).fit_transform(y_all)
else:
    transformed = TSNE(n_components=2).fit_transform(y_all)

# Rebuild structure
yy = []
idx=0
for a in range(num_agents):
    y = []
    for i in range(len(behaviors[a])):
        y.append(transformed[idx])
        idx += 1
    yy.append(y)

print("Max fitness: ", max_fit)
plot_pca(yy, fitnesses, max_fit=max_fit)