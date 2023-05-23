import pandas as pd

train_url = "https://raw.githubusercontent.com/middlebury-csci-0451/CSCI-0451/main/data/palmer-penguins/train.csv"
train = pd.read_csv(train_url)

import seaborn as sns

# Plot the distribution of Culmen Lengths for each penguin species
sns.histplot(data=train, x="Culmen Length (mm)", hue="Species", kde=True, bins=20)


