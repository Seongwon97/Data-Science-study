import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


df =pd.DataFrame({
    'x1': np.random.normal (0, 2,10000),
    'x2': np.random.normal (5, 3,10000),
    'x3': np.random.normal (5, 5, 10000)
})

scaler = preprocessing.RobustScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['x1','x2','x3'])

fig, (ax1, ax2) = plt.subplots(ncols =2, figsize=(6,5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'],ax=ax1)
sns.kdeplot(df['x2'],ax=ax1)
sns.kdeplot(df['x3'],ax=ax1)
ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)

plt.show()
