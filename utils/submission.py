import pandas as pd
import ast

# 1) Load your submission
path = '/srv/scratch1/swallace/CancerSeg/submission.csv'
df   = pd.read_csv(path)

# 2) Parse each string "[p0, p1]" to a Python list, then take the p1 entry
df['Diagnosis'] = df['Diagnosis']\
    .apply(ast.literal_eval)\
    .apply(lambda probs: probs[0])

# 4) Save out the cleaned submission
df.to_csv('submission_probs_class_0.csv', index=False)

# 5) Inspect
print(df.head())