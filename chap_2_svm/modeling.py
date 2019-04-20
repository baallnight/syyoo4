import pandas as pd
import numpy as np
import tensorflow as tf
from numba.testing.ddt import feed_data

ctx = '../data/'
csv = pd.read_csv(ctx+'bmi.csv')

# 데이터 정규화
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100
# one - hot encoding
# thin = (1, 0, 0)
# nortma = (0, 1, 0)
# fat  = (0, 0, 1)

bclass = {"thin": [1, 0, 0], "normal": [0, 1, 0], "fat": [0, 0, 1]}

csv['label_pat'] = csv['label'].apply(lambda x : np.array(bclass[x]))

test_csv = csv[15000:20000]
test_pat = test_csv[['weight', 'height']]
test_ans = list(test_csv['label_pat'])

# ***
# 모델링
# ***

x = tf.placeholder(tf.float32, [None, 2]) # 키 몸무게
y_ = tf.placeholder(tf.float32, [None, 3]) # 레이블 추가
W = tf.Variable(tf.zeros([2, 3]))
b = tf.Variable(tf.zeros([3]))
# 소프트 맥스 회귀 정의
y = tf.nn.softmax(tf.matmul(x, W) + b)

# ***
# 러닝
# ***
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #예측률
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32)) #정확도

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[['weight', 'height']]
    y_ans = list(rows['label_pat'])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print("step= ", step, "cre = ", cre, "acc = ", acc)

#최종 정답률
acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})

print("정답률 : ", acc)
# 정답률 : 0.956