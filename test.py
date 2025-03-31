import tensorflow as tf
import tensorflow as tf
import time

# 模拟输入数据
x = tf.random.normal([1000, 1000])

# 普通函数（eager mode）


def eager_fn(x):
    for _ in range(10):
        x = tf.matmul(x, tf.transpose(x))
        x = tf.nn.relu(x)
    return x

# 编译为图函数


@tf.function
def graph_fn(x):
    for _ in range(10):
        x = tf.matmul(x, tf.transpose(x))
        x = tf.nn.relu(x)
    return x


# warm-up（避免首次调用包含图编译时间）
_ = graph_fn(x)

# 测试 eager 函数运行时间
start = time.time()
_ = eager_fn(x)
eager_time = time.time() - start
print(f"Eager execution time: {eager_time:.4f} seconds")

# 测试 tf.function 编译后的函数运行时间
start = time.time()
_ = graph_fn(x)
graph_time = time.time() - start
print(f"tf.function execution time: {graph_time:.4f} seconds")

# 计算加速比
speedup = eager_time / graph_time
print(f"Speedup: {speedup:.2f}x")
