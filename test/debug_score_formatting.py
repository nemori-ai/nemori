"""
调试分数格式化问题
"""

import numpy as np

# 测试数据
scores = [0.7425, 0.7375, 0.7089, 0.6773, 0.6592]

print("原始分数:")
for i, score in enumerate(scores):
    print(f"  #{i+1}: {score}")

print("\nround(score, 2):")
for i, score in enumerate(scores):
    print(f"  #{i+1}: {round(score, 2)}")

print("\nfloat(round(score, 2)):")
for i, score in enumerate(scores):
    print(f"  #{i+1}: {float(round(score, 2))}")

# 特别测试0.7425和0.7375
print("\n特殊案例:")
print(f"0.7425 -> round(0.7425, 2) = {round(0.7425, 2)}")
print(f"0.7375 -> round(0.7375, 2) = {round(0.7375, 2)}")

# 测试numpy的round
print("\nnp.round:")
arr = np.array(scores)
rounded = np.round(arr, 2)
for i, (orig, rnd) in enumerate(zip(scores, rounded)):
    print(f"  #{i+1}: {orig} -> {rnd}")
