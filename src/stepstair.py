# 계단 오르내리기


import random
import matplotlib.pyplot as plt
import numpy as np

position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
# cumsum은 random하게 생성된 값들의 누적 -> 정규화된 난수를 누적그래프로
walk = steps.cumsum()

print(walk.min())
print(walk.max())

# plot 으로 그래프 모양을 만들고, 첫 번째 파라미터에는 배열을, 두 번째 파라미터에는 그래프 모양으로
plt.plot(walk[:100], 'g+')
# +, --(점선), . 등의 모양을 지정 가능 g(그린), r(레드), b(블루)
plt.title('Step')
plt.show()

# 그래프 저장 plt.savefig()
# plt.savefig("saved_fig.svg")

# argmax() 는 불리언 배열에서 최댓값의 처음 색인을 반환 -> 배열 전체를 모두 확인한다
print((np.abs(walk) >= 10).argmax())  # walk 에서 True 가 최댓값이다
