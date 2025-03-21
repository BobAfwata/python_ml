# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math

# 데이터 준비
x = np.arange(0, 5*np.pi, 0.1) # 0에서 6까지 0.1 간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # cos 함수는 점선으로 그리기
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title('Graph of Sine and Cos')
plt.legend()
plt.show()
