import glob
import matplotlib.pyplot as plt
import brewer2mpl
import math
import numpy as np
from time import sleep

fontsize = 14

skip_main_plot = False
skip_event_plot = False
single = False
single_title = "deathmatch"

plt.style.use('ggplot')

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors


def prettify(text):
    return text.replace("_", " ").title()


def load(dir, event_log=False):
    f_list = glob.glob(dir + '*.log')
    if event_log:
        f_list = glob.glob(dir + '*.eventlog')
    all_data = []
    for filename in f_list:
        print(filename)
        title = filename.split(".")[0]
        data = []
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                d = line.strip().split(",")
                data.append(np.array(d).astype(float))
        all_data.append([title, data ])
    return all_data


def roundup(x, n):
    return int(math.ceil(x / (n+0.0))) * n


def plot_rewards(fig, data):
    for i in range(len(data)):
        x = []
        y = []
        yi = []
        event = False
        title = data[i][0].split("vizdoom_")[1]
        title = title.replace("-2", "")
        if "_event" in title:
            event = True
            title = title.split("_event")[0]
        print(title)
        scale = 1

        if event and title in ["my_way_home"]:
            scale = 100  # Scaling was fixed, so not needed in new experiments

        #if event and title in ["deadly_corridor"]:
            #scale = 0.1

        for d in data[i][1]:
            x.append(d[1])
            y.append(d[2] * 100 * scale)
            yi.append(d[4] * 100 * scale)

        # ignore first data point
        x = x[1:]
        y = y[1:]
        yi = yi[1:]

        # smoothen
        smooth = 25
        x_max = np.max(x)

        if len(x) == 0:
            continue

        y_limits_set = False
        show_noise = True

        if title == "health_gathering":
            idx = 1
            y_max = 2500
            y_min = 0
            y_limits_set = True
        elif title == "health_gathering_supreme":
            idx = 2
            y_max = 1200
            y_min = 0
            y_limits_set = True
        elif title == "deadly_corridor":
            idx = 3
            y_max = 70
            y_min = -1
            y_limits_set = True
        elif title == "my_way_home":
            idx = 4
            y_max = 100
            y_min = -15
            y_limits_set = True
        elif title == "deathmatch":
            idx = 5
            x_max = 75000000
            y_max = 5000
            y_min = 0
            y_limits_set = True
        else:
            continue

        if single and single_title != title:
            continue

        xs = []
        ys = []
        yis = []
        for i in np.arange(0, len(x)):
            mean_x = []
            mean_y = []
            mean_yi = []
            if i == 0 or smooth == 1:
                mean_x.append(x[i])
                mean_y.append(y[i])
                mean_yi.append(yi[i])
            for j in np.arange(max(0, i - round(smooth/2)), min(i + round(smooth/2), len(x))):
                mean_x.append(x[j])
                mean_y.append(y[j])
                mean_yi.append(yi[j])
            xs.append(np.mean(mean_x))
            ys.append(np.mean(mean_y))
            yis.append(np.mean(mean_yi))

        if not single:
            ax = plt.subplot(1, 5, idx)
            if idx > 5:
                plt.xlabel('Time step')
            if idx % 5 == 1:
                plt.ylabel('Reward / Episode', fontsize=fontsize)
            if title == "deathmatch":
                plt.xticks(np.arange(0, x_max+1, 25000000))
            else:
                plt.xticks(np.arange(0, x_max+x_max*0.1, 2000000))
        else:
            ax = plt.subplot(1, 1, 1)

        if event:
            if not y_limits_set:
                y_min = min(0, np.min(ys) - np.min(ys) * 0.1)
                y_max = np.max(ys) + np.max(ys) * 0.1
            plt.xlim(0, x_max)
            plt.ylim(y_min, y_max)

        color = '#1f77b4'
        if event:
            color = '#d62728'

        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
        label = "A2C" if not event else "A2C+RoE"
        l = ax.plot(xs, ys, color=color, linewidth=1.0, label=label)
        if show_noise:
            ax.plot(x, y, linewidth=1, alpha=0.1, color=color)
        plt.title(prettify(title), fontsize=fontsize)

        if title == "deadly_corridor" or single:
            if event:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=fontsize)

    if single:
        plt.savefig('rewards_single.pdf')
    else:
        plt.savefig('rewards.pdf')


def plot_events(fig, data_event, reward_value=False, clip=0.01):

    for i in range(len(data_event)):

        event = True if "_event" in data_event[i][0] else False

        x = []
        movement = []
        health = []
        armor = []
        shoot = []
        ammo = []
        weapon0 = []
        weapon1 = []
        weapon2 = []
        weapon3 = []
        weapon4 = []
        weapon5 = []
        weapon6 = []
        weapon7 = []
        weapon8 = []
        weapon9 = []
        weapon0kill = []
        weapon1kill = []
        weapon2kill = []
        weapon3kill = []
        weapon4kill = []
        weapon5kill = []
        weapon6kill = []
        weapon7kill = []
        weapon8kill = []
        weapon9kill = []
        kill = []

        if reward_value:
            for d in data_event[i][1]:
                x.append(d[0])
                movement.append(1/max(clip, d[1]))
                health.append(1/max(clip, d[2]))
                armor.append(1/max(clip, d[3]))
                shoot.append(1/max(clip, d[4]))
                ammo.append(1/max(clip, d[5]))
                weapon0.append(1/max(clip, d[6]))
                weapon1.append(1/max(clip, d[7]))
                weapon2.append(1/max(clip, d[8]))
                weapon3.append(1/max(clip, d[9]))
                weapon4.append(1/max(clip, d[10]))
                weapon5.append(1/max(clip, d[11]))
                weapon6.append(1/max(clip, d[12]))
                weapon7.append(1/max(clip, d[13]))
                weapon8.append(1/max(clip, d[14]))
                weapon9.append(1/max(clip, d[15]))

                if len(d) > 17:
                    weapon0kill.append(1 / max(clip, d[17]))
                    weapon1kill.append(1 / max(clip, d[18]))
                    weapon2kill.append(1 / max(clip, d[19]))
                    weapon3kill.append(1 / max(clip, d[20]))
                    weapon4kill.append(1 / max(clip, d[21]))
                    weapon5kill.append(1 / max(clip, d[22]))
                    weapon6kill.append(1 / max(clip, d[23]))
                    weapon7kill.append(1 / max(clip, d[24]))
                    weapon8kill.append(1 / max(clip, d[25]))
                    weapon9kill.append(1 / max(clip, d[26]))

                kill.append(1/max(clip, d[16]))
        else:
            for d in data_event[i][1]:
                x.append(d[0])
                movement.append(d[1])
                health.append(d[2])
                armor.append(d[3])
                shoot.append(d[4])
                ammo.append(d[5])

                weapon0.append(d[6])
                weapon1.append(d[7])
                weapon2.append(d[8])
                weapon3.append(d[9])
                weapon4.append(d[10])
                weapon5.append(d[11])
                weapon6.append(d[12])
                weapon7.append(d[13])
                weapon8.append(d[14])
                weapon9.append(d[15])

                if len(d) > 17:
                    weapon0kill.append(d[17])
                    weapon1kill.append(d[18])
                    weapon2kill.append(d[19])
                    weapon3kill.append(d[20])
                    weapon4kill.append(d[21])
                    weapon5kill.append(d[22])
                    weapon6kill.append(d[23])
                    weapon7kill.append(d[24])
                    weapon8kill.append(d[25])
                    weapon9kill.append(d[26])

                kill.append(d[16])

        if event:
            title = data_event[i][0].split("vizdoom_")[1].split("_event")[0]
        else:
            title = data_event[i][0].split("vizdoom_")[1]

        title = title.replace("-2", "")

        x_max = np.max(x)

        if title == "health_gathering":
            idx = 1
            y_max = 100
        elif title == "health_gathering_supreme":
            idx = 2
            y_max = 60
        elif title == "deadly_corridor":
            idx = 3
            y_max = 40
        elif title == "my_way_home":
            idx = 4
            y_max = 25
        elif title == "deathmatch":
            idx = 5
            if not single:
                y_max = 400
            x_max = 75000000
        else:
            continue

        if event:
            idx += 5

        if single and single_title != title:
            continue

        if not single:
            ax = plt.subplot(2, 5, idx)
            if idx > 5:
                plt.xlabel('Time step')
            if idx % 5 == 1:
                if event:
                    plt.ylabel(r"A2C+RoE" + "\n" + "Events / Episode", fontsize=fontsize)
                else:
                    plt.ylabel(r"A2C" + "\n" + "Events / Episode", fontsize=fontsize)
            if title == "deathmatch":
                plt.xticks(np.arange(0, x_max+1, 25000000))
            else:
                plt.xticks(np.arange(0, x_max+x_max*0.1, 2000000))
        else:
            ax1 = plt.subplot(1, 1, 1)
            plt.xlim(0, x_max)
            plt.title("Common Events", fontsize=fontsize)
            ax2 = plt.subplot(2, 1, 1)
            plt.xlim(0, x_max)
            plt.title("Weapon Pickup Events", fontsize=fontsize)
            ax3 = plt.subplot(2, 1, 2)
            plt.xlim(0, x_max)
            plt.title("Kills", fontsize=fontsize)


        if single:
            ax = ax1

        l0 = ax.plot(x, kill, label='Kills', linewidth=1)
        l0 = ax.plot(x, movement, label='Moves', linewidth=1)
        l1 = ax.plot(x, health, label='Medkit pickups', linewidth=1, color="green")
        l2 = ax.plot(x, armor, label='Armor pickups', linewidth=1, color="purple")
        l3 = ax.plot(x, shoot, label='Shots', linewidth=1, color="orange")
        #l4 = ax.plot(x, ammo, label='Ammo')

        if single:
            ax = ax2
            l5 = ax.plot(x, weapon0, label='W. 0 pickup')
            l6 = ax.plot(x, weapon1, label='W. 1 pickup')
            l7 = ax.plot(x, weapon2, label='W. 2 pickup')
            l8 = ax.plot(x, weapon3, label='W. 3 pickup')
            l9 = ax.plot(x, weapon4, label='W. 4 pickup')
            l10 = ax.plot(x, weapon5, label='W. 5 pickup')
            l11 = ax.plot(x, weapon6, label='W. 6 pickup')
            l12 = ax.plot(x, weapon7, label='W. 7 pickup')
            l13 = ax.plot(x, weapon8, label='W. 8 pickup')
            l14 = ax.plot(x, weapon9, label='W. 9 pickup')

        if single:
            ax = ax3

        if len(d) > 16 and single:
            l15 = ax.plot(x, weapon0kill, label='W. 0 kill')
            l16 = ax.plot(x, weapon1kill, label='W. 1 kill')
            l17 = ax.plot(x, weapon2kill, label='W. 2 kill')
            l18 = ax.plot(x, weapon3kill, label='W. 3 kill')
            l19 = ax.plot(x, weapon4kill, label='W. 4 kill')
            l20 = ax.plot(x, weapon5kill, label='W. 5 kill')
            l21 = ax.plot(x, weapon6kill, label='W. 6 kill')
            l22 = ax.plot(x, weapon7kill, label='W. 7 kill')
            l23 = ax.plot(x, weapon8kill, label='W. 8 kill')
            l24 = ax.plot(x, weapon9kill, label='W. 9 kill')

        if not single:
            plt.title(prettify(title), fontsize=fontsize)

        if title == "deadly_corridor" or single:
            if event:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=fontsize)

        plt.xlim(0, x_max)
        if not single:
            plt.ylim(0, y_max)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))

    if reward_value:
        plt.savefig('event_rewards.pdf')
    else:
        if single:
            plt.savefig('events_single.pdf')
        else:
            plt.savefig('events.pdf')

dir = 'log/'
data = load(dir)
data_event = load(dir, event_log=True)
fig = plt.figure(figsize=(16, 4))
plt.subplots_adjust(left=0.045, right=0.985, top=0.75, bottom=0.15, wspace=0.2, hspace=0.2)
event_plot_data = []

if not skip_main_plot:
    plot_rewards(fig, data)

if not skip_event_plot:
    if not single:
        fig = plt.figure(figsize=(16, 6))
        plt.subplots_adjust(left=0.04, right=0.985, top=0.85, bottom=0.1, wspace=0.2, hspace=0.4)
        plot_events(fig, data_event)
    else:
        fig = plt.figure(figsize=(6, 6))
        plt.subplots_adjust(left=0.04, right=0.985, top=0.8, bottom=0.1, wspace=0.2, hspace=0.4)
        plot_events(fig, data_event)