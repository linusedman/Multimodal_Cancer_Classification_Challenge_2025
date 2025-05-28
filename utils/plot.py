import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = '/srv/scratch1/swallace/CancerSeg/SWINv2_BF_log.csv'

df = pd.read_csv(path)

print(df.head())

train_loss = df['train_loss']
train_acc = df['train_acc']
val_loss = df['val_loss']
val_acc = df['val_acc']
epoch = df['epoch']


plt.plot(epoch, train_loss, label = "Train Loss")
plt.plot(epoch, val_loss, label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.savefig("Train_curve.png")