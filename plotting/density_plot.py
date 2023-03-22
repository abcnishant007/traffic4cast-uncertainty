#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Read files: format from Alex:
# speed:  [0, 2, 4, 6]
# volume: [1, 3, 5, 7]
#  (samples, 4, 495, 436, 8), where in second dim:
#         0: ground truth,
#         1: point prediction,
#         2: epistemic uncertainty,
#         4: aleatoric uncertainty

import os
import h5py
from tqdm import tqdm
from smartprint import smartprint as sprint


import matplotlib

matplotlib.use("TkAgg")

model_list = ["unet", "unetpp"]
foldername_prefix = "h5_files_"
city_list = ["moscow", "antwerp"]

for model in model_list:
    for city in city_list:
        filename = os.path.join(foldername_prefix + model, "pred_combo_" + city + ".h5")

        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])
            sprint(len(data))

            for i in tqdm(range(len(data)), desc="Reading " + model + city):
                sample = data[i]
                sprint(sample.shape)
                break


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def single_plot(a, y_list, x_list, combine="mean", y_pred_name="GT", y_varname="Volume"):
    """
    a : numpy array of shape (4, 495, 436, 8)
    combine : ["mean", "mean_of_square"] # how to combine 4 channels
    y_list : list of y channels for plot;
    x_list : list of x channels for plot; Both x_list and y_list are from the the fourth dimension (index=3)
    thresh_filter_2D : 0  # 2D np.where output above threshold
    """
    chan = {"GT": 0, "PP": 1, "EU": 2, "AU": 3}
    speed_channels = [0, 2, 4, 6]

    density_vals = []
    non_zero_indices_in_density_channels = []
    for index, i in enumerate(x_list):

        # we need to filter zeros in speed channels to avoid np.nans in the density computation
        where_non_zero = a[0, :, :, speed_channels[index]] > 0
        x = a[0, :, :, i][where_non_zero]

        non_zero_indices_in_density_channels.append(where_non_zero)

        if combine == "mean":
            density_vals.extend(x.flatten().tolist())

    y = []
    for index, i in enumerate(y_list):
        x = a[chan[y_pred_name], :, :, i][non_zero_indices_in_density_channels[index]]
        if combine == "mean":
            y.extend(x.flatten().tolist())

    assert len(density_vals) == len(y)
    return density_vals, y


for model in model_list[0:1]:
    for city in city_list[0:1]:
        filename = os.path.join(foldername_prefix + model, "pred_combo_" + city + ".h5")

        X = []
        Y = []
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])
            sprint(len(data))

            density_proxy = []
            for i in tqdm(range(len(data)), desc="reading all files for one city"):
                sample = data[i]

                ss = sample.shape
                sample_with_density = np.random.rand(ss[0], ss[1], ss[2], 8 + 4)
                sample_with_density[:, :, :, :8] = np.array(sample)

                for speed_channel in [0, 2, 4, 6]:
                    vol_channel = speed_channel + 1

                    # 0 in the first dimension is to because we don't care about the densities computed
                    # from anything apart from GT channels for speed and volume, all others are set to np.nan
                    sample_with_density[0, :, :, 8 + speed_channel // 2] = (
                        sample[0, :, :, vol_channel] / sample[0, :, :, speed_channel]
                    )

                    for j in range(1, 4):
                        sample_with_density[j, :, :, 8 + speed_channel // 2] = np.nan

                x, y = single_plot(
                    sample_with_density,
                    x_list=[8, 9, 10, 11],
                    y_list=[0, 2, 4, 6],
                    combine="mean",
                    y_pred_name="AU",
                    y_varname="Speed",
                )

                X.extend(x)
                Y.extend(y)

            ss = sorted(zip(X, Y))
            X = [x for x, y in ss]
            Y = [y for x, y in ss]
            plt.plot(X, label="density sorted")
            plt.plot(Y, label="Y")
            plt.yscale("log")
            plt.legend()
            plt.show()

            plt.plot(X, label="density sorted")
            plt.plot(Y, label="Y")
            plt.legend()
            plt.show()
