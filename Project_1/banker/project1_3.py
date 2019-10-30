import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pd.read_csv('../data/german.data', sep=' ',
                     names=features+[target]) 

#sns.distplot(df["age"], kde=False)
#plt.show()

df.sort_values("age", inplace=True)

split = int(len(df)/3)
young = df.iloc[:split]
middle = df.iloc[split:2*split]
old = df.iloc[2*split:]

sns.distplot(young["age"], kde=False)
plt.show()
plt.clf()
sns.distplot(middle["age"], kde=False)
plt.show()
plt.clf()
sns.distplot(old["age"], kde=False)
plt.show()