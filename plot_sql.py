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
import imageio

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
exp_id = 1

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

# print(data)
'''
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
        yhat = savgol_filter(y, 5001, 3)  # window size 5001, polynomial order 3
        plt.plot(x, yhat, color=colors[a])
    plt.savefig(f'plots/events_{exp_id}_{name}')
    plt.clf()
'''

smooth_over = 1000
'''
smoothed_agent_episodic_events = []
for a in range(num_agents):
    episodic_events = []
    for e in data[a]:
        episodic_events.append(e)
    smoothed_episodic_events = []
    smoothed = []
    for i in range(len(episodic_events)):
        smoothed.append(episodic_events[i])
        if len(smoothed) >= smooth_over:
            smoothed_episodic_events.append(np.mean(smoothed, axis=0))
            smoothed.clear()
    smoothed_agent_episodic_events.append(smoothed_episodic_events)


def plot_pca(data, points, step, pca=True):
    fig, plot = plt.subplots()
    fig.set_size_inches(4, 4)
    plt.prism()
    plt.plot(data[:, 0], data[:, 1], 'o', markerfacecolor='grey', markersize=1, fillstyle='full', markeredgewidth=0.0)
    #colors = ['red', 'blue', 'green', 'purple', 'orange', 'teal', 'black', 'grey']
    for i in range(len(points)):
        plt.plot(points[i][0], points[i][1], 'o', markerfacecolor=colors[i], markersize=6, fillstyle='full', markeredgewidth=0.0)
    plot.set_xticks(())
    plot.set_yticks(())
    plt.title(str(step))
    plt.tight_layout(pad=-0.5, w_pad=-0.5, h_pad=-0.5)
    #fig.savefig("plots/{}.pdf".format("pca" if pca else "t-sne"), bbox_inches='tight', pad_inches=0)
    fig.savefig("plots/pca/{}_step_{}.png".format("pca" if pca else "t-sne", step), bbox_inches='tight', pad_inches=0)
    return fig

y_all = []
xx = []
yy = []
for a in range(num_agents):
    x = []
    y = []
    for i in range(len(smoothed_agent_episodic_events[a])):
        events = smoothed_agent_episodic_events[a][i]
        x.append(events[0])
        y.append(events[1:])
        y_all.append(events[1:])
    xx.append(x)
    yy.append(y)

transformed_pca = PCA(n_components=2).fit_transform(y_all)
#transformed_pca = [(events[2], events[10]) for events in y_all]
#transformed_pca = np.array(transformed_pca)
#print(transformed_pca)

for i in range(len(xx[0])):
    step = xx[0][i]
    points = []
    for a in range(num_agents):
        point = yy[a][i]
        for i in range(len(y_all)):
            y = y_all[i]
            if np.array_equal(y, point):
                points.append(transformed_pca[i])
                break
    assert len(points) == num_agents

    plot_pca(transformed_pca, points, step)
'''
images = []
filenames = glob.glob('plots/pca/pca_*.png')
d = {}
for filename in filenames:
    step = int(filename.split("_")[-1].split(".png")[0])
    d[step] = filename

for sted in sorted(d.keys):
    filename = d[key]
    images.append(imageio.imread(filename))
imageio.mimsave(f'plots/pca/pca_armor-5_{smooth_over}.gif', images)
# transformed_tsne = TSNE(n_components=2).fit_transform(y_all)