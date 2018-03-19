import glob
import matplotlib.pyplot as plt
import brewer2mpl
import math
import numpy as np
from time import sleep
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('ggplot')

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors


def load_data(scenario, event):
    filename = "./heat_data/" + scenario + ".p"
    if event:
        filename = "./heat_data/" + scenario + "_event.p"

    data = pickle.load(open(filename, "rb"))

    map_size = 1536
    grids = 32
    grid_size = map_size / grids

    max_x = -100000
    min_x = 100000
    max_y = -100000
    min_y = 100000

    grid = []
    for episode in data:
        grid_episode = np.zeros((grids+2, grids+2))
        for position in episode:
            x = position[0] + 256
            y = position[1] + 256
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
            xx = (int)(x / grid_size) + 1
            yy = (int)(y / grid_size) + 1
            grid_episode[xx][yy] += 1
        grid_episode = np.divide(grid_episode, np.full(grid_episode.shape, len(episode)))
        grid.append(grid_episode)

    mean = np.mean(grid, axis=0)
    #log = np.log(mean)
    return mean

scenario = "deathmatch"
event_based = load_data(scenario, True)
baseline = load_data(scenario, False)

event_based_chainsaw = load_data(scenario + "-chainsaw", True)
baseline_chainsaw = load_data(scenario + "-chainsaw", False)

event_based_chaingun = load_data(scenario + "-chaingun", True)
baseline_chaingun = load_data(scenario + "-chaingun", False)

event_based_shotgun = load_data(scenario + "-shotgun", True)
baseline_shotgun = load_data(scenario + "-shotgun", False)

event_based_plasma = load_data(scenario + "-plasma", True)
baseline_plasma = load_data(scenario + "-plasma", False)

event_based_rocket = load_data(scenario + "-rocket", True)
baseline_rocket = load_data(scenario + "-rocket", False)

# Set up figure and image grid
fig = plt.figure(figsize=(11, 3.75))
plt.subplots_adjust(left=0.02, right=0.905, top=0.9, bottom=0.05, wspace=0.1, hspace=0.1)

cmap = 'plasma'
cmap = 'inferno'
cmap = 'magma'
cmap = 'viridis'

fontsize=10


def plot(data, idx, title="", algo=""):
    ax = plt.subplot(2, 6, idx)
    img = ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=0, vmax=0.025)
    ax.grid(linewidth=0)
    ax.set_title(title, fontsize=fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel(algo, fontsize=fontsize)
    return img

plot(baseline, 1, "Deathmatch", "A2C")
plot(baseline_chainsaw, 2, "DM Chainsaw", "")
plot(baseline_shotgun, 3, "DM Shotgun", "")
plot(baseline_chaingun, 4, "DM Chaingun", "")
plot(baseline_rocket, 5, "DM Rocket", "")
plot(baseline_plasma, 6, "DM Plasma", "")

plot(event_based, 7, "", "A2C+RoE")
plot(event_based_chainsaw, 8, "", "")
plot(event_based_shotgun, 9, "", "")
plot(event_based_chaingun, 10, "", "")
plot(event_based_rocket, 11, "", "")
img = plot(event_based_plasma, 12, "", "")

cbar_ax = fig.add_axes([0.915, 0.051, 0.03, 0.848])
cbar = fig.colorbar(img, cax=cbar_ax)
cbar.ax.tick_params(labelsize=fontsize)

plt.draw()
#plt.show()
plt.savefig('heat_map_' + scenario + '.pdf')