import pandas as pd
import ast
import torch
import torch.nn.functional as F

# 1) Load the CSV
df = pd.read_csv('/srv/scratch1/swallace/CancerSeg/submission_logits.csv')

# 2) Parse the 'logits' column from strings like "[x, y]" into real lists
df['logits_list'] = df['logits'].apply(ast.literal_eval)

# 3) Turn that into a tensor of shape [N, 2] and compute softmax along dim=1
logits_tensor = torch.tensor(df['logits_list'].tolist())    # shape: [N,2]
probs_tensor  = F.softmax(logits_tensor, dim=1)              # shape: [N,2]

# 4) Extract the P(class=1) column
df['p_cancer'] = probs_tensor[:, 1].numpy()

# (Optional) drop the intermediate lists column if you like
df = df.drop(columns=['logits_list'])
df = df.drop(columns=['logits'])
df.rename(columns={
    'filename': 'Name',
    'p_cancer': 'Diagnosis'
}, inplace=True)

# 5) Save back to CSV
df.to_csv('submission_with_probs.csv', index=False)

print(df.head())