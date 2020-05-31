import matplotlib.pyplot as plt
import numpy as np
f = open('accuracy.txt')
f2= open('loss.txt')
i=0
fig_loss = np.zeros([200])
fig_accuracy = np.zeros([200])
for line in f:
    if(i%100==0):
     line = line.strip('\n')
     line = line.split(' ')
     fig_accuracy[int(i/100)]=float(line[0])
    i=i+1
i=0
for line in f2:
    if(i%100==0):
     line = line.strip('\n')
     line = line.split(' ')
     fig_loss[int(i/100)]=float(line[0])
    i=i+1
f.close
f2.close

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.plot(np.arange(200), fig_loss, label="Loss")
# 按一定间隔显示实现方法
# ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
lns2 = ax2.plot(np.arange(200), fig_accuracy, 'r', label="Accuracy")
ax1.set_xlabel('iteration')
ax1.set_ylabel('training loss')
ax2.set_ylabel('training accuracy')
# 合并图例
lns = lns1 + lns2
labels = ["Loss", "Accuracy"]
# labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=7)
plt.show()