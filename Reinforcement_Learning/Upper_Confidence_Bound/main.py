# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB

"""
Step 1: 
At each round n, we consider two numbers for each ad i:
• Ni(n) - the number of times the ad i was selected up to round n,
• Ri(n) - the sum of rewards of the ad i up to round n.
"""

total_users = len(dataset)
total_Ads = len(dataset.values[0, ])
ads_selected = []
list_totalAds_selected = [0] * total_Ads
list_Sum_of_rewards = [0] * total_Ads
total_rewards = 0

"""
Step 2:
From these two numbers we compute:
• the average reward of ad i up to round n
ři(n) = Ri(n) / Ni(n)
• the confidence interval [ři(n) – ∆i(n), ři(n) + ∆i(n)] at round n with
∆i(n) = sqrt (3/2 * (log(n) / Ni(n))
"""

for n in range(total_users):
    ad = 0
    max_UCB = 0
    for i in range(total_Ads):
        if list_totalAds_selected[i] > 0:
            average_reward = list_Sum_of_rewards[i] / list_totalAds_selected[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / list_totalAds_selected[i])
            """
            Step 3:
            We select the ad i that has the maximum UCB = ři(n) + ∆i(n).
            """
            UCB = average_reward + delta_i
        else:
            UCB = 1e400

        if UCB > max_UCB:
            max_UCB = UCB
            ad = i
    ads_selected.append(ad)
    list_totalAds_selected[ad] += 1
    reward = dataset.values[n, ad]
    list_Sum_of_rewards[ad] += reward
    total_rewards += reward

# Visualization
plt.hist(ads_selected)
plt.title("Histogram for Ads selected")
plt.xlabel('Ads')
plt.ylabel('y')
plt.show()
