import tensorflow as tf
import numpy as np

# [털, 날개]

x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)
y_data = np.array(
   [[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]]
)
"""
[기타, 포유류, 조류] 의 인덱스 값은 0, 1, 2
이것을 원핫인코딩으로 만들면
[털, 날개] -> [기타, 포유류, 조류]
[0, 0] -> [1, 0, 0]
[1, 0] -> [0, 1, 0]
[1, 1] -> [0, 0, 1] 조류

"""


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 3], -1, 1)) # -1은 all
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)
# nn 은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3]

model = tf.nn.softmax(L)
print(model)

# Tensor("Softmax:0", shape=(?, 3), dtype=float32)
"""
softmax() 를 사용하여 출력값을 사용하기 쉽게 만듦
소프트맥스 합수는 결과값을 전체합이 1인 확률로 만들어 주는 함수
예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23]
"""

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

#***
#러닝
#****
#tf.initialize_all_variables().run()
#init = tf.global_variables_initializer()

#train_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in range(100):
    sess.run(train_op, {X: x_data, Y: y_data})

"""
tf.argmax 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옴
예) [[0, 1, 0][1, 0, 0] -> [1, 0]
[[0.2, 0.7, 0.1][0.9, 0.1, 0.]] -> [1, 0]
"""
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값', sess.run(prediction, {X: x_data}))
print('측정값', sess.run(target, {Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))
