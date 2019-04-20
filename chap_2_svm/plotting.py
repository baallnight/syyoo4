import matplotlib.pyplot as plt
import pandas as pd

ctx = '../data/'

csv = pd.read_csv(ctx+'bmi.csv', index_col=2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# 서브 플롯 전용 ... 지정한 레이블을 임의의 색으로 칠하기

def scatter(lbl, color):
    b = csv.loc[lbl]
    ax.scatter(b['weight'], b['height'], c= color, label= lbl)

scatter('fat', 'red')
scatter('normal', 'green')
scatter('thin', 'yellow')

ax.legend()
plt.savefig(ctx+'bmi-test.png')
plt.show()
