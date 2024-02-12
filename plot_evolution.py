import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import glob

path = './results/*.csv'

def collect_data(path):
    files = glob.glob(path)
    files = sorted(files, key=lambda x:int(x.split('timeframe')[1].split('.')[0]))
    times = np.zeros(len(files))
    volume = np.zeros(len(files))
    x = []
    h = []
    p = []
    K = []
    
    for i in range(len(files)):
        data = pd.read_csv(files[i], sep='\t', skiprows=2).to_numpy()
        x.append(data[:, 0])
        h.append(data[:, 1])
        p.append(data[:, 2])
        K.append(data[:, 3])

        temp = pd.read_csv(files[i])
        times[i] = float(temp.keys()[0].split('=')[1])
        volume[i] = float(temp[temp.keys()[0]][0].split('=')[1])

    return times, volume, x, h, p, K

times, vol, x, h, p, K = collect_data(path)

fig, axs = plt.subplots(1, 2)
cm = mpl.cm.viridis(np.linspace(0, 1, len(x)//5+1))
for i in range(0, len(x), 5):
    axs[0].plot(x[i], h[i], c=cm[i//5])
    axs[1].plot(x[i], K[i], c=cm[i//5])

fig.show()