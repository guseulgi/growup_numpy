import matplotlib.pyplot as plt
import numpy as np

# matplotlib 는 2D 그래프를 위한 데스크톱 패키지로 출판물 수준의 그래프를 만들 수 있다
# 모든 운영체제의 GUI 백엔드를 지원
# pdf, svg, jpg, png, bmp, gif 등 일반적으로 널리 사용되는 벡터 포맷과 래스터 포맷으로 그래프를 저장할 수 있다

# 1차 그래프
data = np.arange(10)
# .plot() 으로 배열을 넣어준다
# plt.plot(data)
# .show() 로 그려준다
# plt.show()

# figure
fig = plt.figure()
# .figure() 자체는 빈 윈도우를 만들어준다 - 특정 배치에 맞춰 여러 개의 서브플롯을 넣어준다
# .add_subplot() 으로 최소 하나 이상의 subplot 을 figure 에 넣어줘야 채워진다
# add_subplot의 첫 번째 파라미터와 두 번째 파라미터는 가로x세로 크기이며 세번째 파라미터는 인덱스를 의미한다(1부터 시작)
ax1 = fig.add_subplot(2, 2, 1)  # 즉 ax1 은 2x2 크기의 fig에서 첫 번째 서브플롯을 선택하겠다는 의미이다
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

# hist() 는 히스토그램 그래프
_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)  # 첫번째 서브플롯
# scatter 는 산개 그래프
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))  # 두번째 서브플롯

# figure 는 plt.show() 로 바로 그려줄 수 있다
# figure 에 만들어진 서브플롯 중 하나에 그래프가 그려진다 (아래의 경우 가장 마지막 서브플롯에 그려짐)
plt.plot(np.random.randn(50).cumsum(), 'k--')
# cumsum() 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산
# plt.show()


# plt.subplots() 배열과 서브플롯 객체를 서로 생성하여 반환
# fig, axes = plt.subplots(2, 3)
# axes 에는 2차원 배열이 들어가짐 sharex, sharey 를 사용하여 x, y축을 가질 수 있다
'''
  pyplot.subplots 옵션
  nrows : 서브플롯의 Row 수
  ncols : 서브플롯의 Column 수
  sharex : 모든 서브플롯이 같은 x축 눈금을 사용
  sharey : 모든 서브플롯이 같은 y축 눈금을 사용
  subplot_kw : add_subplot 을 사용하여 서브플롯을 생성할 때 사용할 키워드를 담고 있는 사전
  fig_kw : figure 를 생성할 때 사용할 추가적인 키워드 인자
'''


# 서브플롯 간격 조절
# matplotlib 은 서브플롯 간에 적당한 간격과 여백을 추가해준다. 이는 전체 그래프의 높이와 너비에 따라 상대적으로 결정된다.
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
# wspace, hspace 는 서브플롯 간격을 위해 각각 figure 의 너비와 높이에 대한 비율을 조절할 수 있다
plt.subplots_adjust(wspace=1, hspace=1)
plt.show()
