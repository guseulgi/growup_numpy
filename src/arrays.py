from random import normalvariate
from numpy.linalg import inv, qr
import matplotlib.pyplot as plt
import numpy as np

'''
  ndarray 는 대규모 데이터 집합을 담을 수 있는 포괄적인 다차원 배열이다.
  데이터의 원소들은 같은 자료형이어야 하며,
  배열의 차원의 크기를 알려주는 shape 라는 튜플과 배열에 저장된 자료형을 알려주는 dtype 객체가 존재한다.
  ndim 의 경우 차원을 반환해준다.
'''


# 배열 생성
# array: 입력 데이터를 ndarray로 변환 -> 입력 데이터는 기본적으로 복사됨
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
# 해당 데이터로부터 형태를 추론하여 2차원 형태로 나오게 된다.
arr2 = np.array(data2)

# print(arr1, '\n', arr2)
# print(arr2.ndim)  # 2
# print(arr1.dtype)  # float64

# zeros / ones
# print(np.zeros(10))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# print(np.zeros((2, 3)))  # [[0. 0. 0.] [0. 0. 0.]]
# print(np.ones(5))  # [1. 1. 1. 1. 1.]

# empty: 0으로 초기화된 배열을 반환하지 않고 가비지 값으로 채워진 배열을 반환한다
# print(np.empty(10))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# full: 인자로 받은 dtype 과 배열의 모양을 생성하고 인자로 받은 값으로 배열을 채운다

# asarray: 입력 데이터를 ndarray로 변환하지만 입력 데이터가 이미 ndarray 일 경우 복사가 일어나진 않는다

# arange: 내장 range 함수와 유사하지만 리스트 대신 ndarray를 반환
# print(np.arange(15))  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

# eye, identity: NxN 크기의 단위행렬을 생성한다. 좌상단 -> 우하단을 잇는 대각선은 1로 채워지고 나머지는 0으로 채워진다


'''
  디스크에서 데이터를 읽고 쓰기 편하도록 하위 레벨의 표현에 직접적으로 맞춰져 있어
  저수준 언어(C, 포트란 등)로 작성된 코드와 쉽게 연동이 가능해진다.
  산술 데이터의 dtype 은 float, int 같은 자료형의 이름과 하나의 원소가 차지하는 비트 수로 이뤄진다.
'''
# astype: 배열의 dtype 을 다른 형으로 명시적 형변환
arr = np.array([1, 2, 3, 4, 5])
# print(arr.dtype)  # int64
float_arr = arr.astype(np.float64)
# print(float_arr.dtype)  # float64

'''
  벡터화
  for 문을 작성하지 않고 데이터를 일괄 처리할 수 있다.
  1. 같은 크기의 배열 간 산술 연산은 배열의 각 원소 단위로 적용된다.
  2. 스칼라 인자가 포함된 산술 연산의 경우 배열 내 모든 원소에 스칼라 인자가 적용된다.
  3. 같은 크기를 가진 배열 간의 비교 연산이 가능하다.
    -> 크기가 다른 배열 간의 연산은 브로드캐스팅이라 한다.
'''

'''
  색인/슬라이싱
  1차원 배열은 리스트와 유사하게 동작한다.
'''
arr3 = np.arange(10)
# print(arr3[5])  # 5
# print(arr3[5:8])  # 5 6 7

# 브로드캐스팅: 배열 조각에 스칼라값을 대입하면 선택 영역 전체로 전파되는 현상
# 이 현상은 리스트와는 달리 원본 배열에 그대로 반영된다는 것이다. (데이터 복사가 일어나지 않음)
arr3[5:8] = 10
# print(arr3)  # [ 0  1  2  3  4 10 10 10  8  9]

arr_slc = arr3[5:8]
# print(arr_slc)  # 10 10 10

arr_slc[1] = 20
# print(arr3)  # [ 0  1  2  3  4 10 20 10  8  9] <- 데이터의 복사가 일어남을 확인 가능
'''
  데이터의 복사가 자주 일어나는 다른 언어를 사용한다면 복사가 일어나지 않는 것이 특이점이다.
  이는 Numpy 의 설계상 대용량 데이터 처리를 염두했기 때문에 데이터 복사를 남발하면 성능과 메모리 문제로 인해 복사가 일어나지 않도록 설계된 것이다.
'''

# 원본 배열을 건들고 싶지 않다면 copy() 를 사용하여 명시적으로 배열 복사가 일어나도록 한다.
arr_slc2 = arr3[5:8].copy()
arr_slc2[2] = 50
# print(arr3)  # [ 0  1  2  3  4 10 20 10  8  9] <- 데이터 복사가 일어나지 않음을 확인 가능

# [:]: 배열의 모든 값을 할당

# 다차원 배열
# 2차원 배열에선 각 색인에 해당하는 요소는 스칼라값이 아닌 1차원 배열이다.
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr2d[2])  # [7 8 9]

# 즉 2차원 배열에서 개별 요소에 접근하기 위해선 재귀적으로 접근해줘야 한다.
# print(arr2d[2][0])  # 7
# ,를 사용하여 재귀적 접근과 동일한 결과를 유추할 수 있다.
# print(arr2d[2, 0])  # 7
# 즉 첫번째 색인을 Row(행)으로, 두번째 색인을 Column(열)로 접근하는 것이다.

# 다차원 배열에서 마지막 색인을 생략하면 반환되는 객체는 상위 차원의 데이터를 포함하고 있는 한 차원 낮은 ndarray 가 된다.
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(arr3d.shape)  # (2, 2, 3)

# print(arr3d[0].shape)  # (2, 3)
# print(arr3d[1, 0])  # [7 8 9]


# 슬라이스
# 축을 따라 선택한 영역 내 요소를 선택
# print(arr2d[:2])  # [[1 2 3] [4 5 6]]
# print(arr2d[:2, 1:])  # [[2 3] [5 6]]
# 슬라이싱을 해서 같은 차원의 배열에 대한 뷰를 얻을 수 있다

# print(arr2d[2])  # [7 8 9]
# print(arr2d[2, :])  # [7 8 9]
# print(arr2d[2:, :])  # [[7 8 9]]


# np.random.randn()
names = np.array(['Bob', 'Joe', 'Bob', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)  # 7 Row, 4 Colum 의 랜덤값
# print(data)

# print(names == 'Bob')  # [ True False True True False False False]

# print(data[names == 'Bob'])  # Boolean 값을 기반으로 Row가 True 인 data Row 가 출력
# 불리언 배열은 반드시 색인하려는 축의 길이와 동일한 길이를 가져야한다
# 물론 배열의 크기가 달라서 실패하지 않는다
# print(~(names == 'Bob'))  # [False  True False False  True  True  True]
# print(data[~(names == 'Bob'), 2:])

# 배열에 불리언 색인을 이용하여 데이터를 선택하면 반환되는 배열의 내용이 바뀌지 않더라도 데이터 복사가 발생한다
# 불리언 배열에서는 and, or 대신 &, | 를 사용한다


'''
  팬시 색인
  정수 배열을 사용한 색인을 설명하기 위한 용어
  슬라이싱과 달리 선택된 데이터를 새로운 배열로 복사한다
'''
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i

# print(arr)
# [[0. 0. 0. 0.] [1. 1. 1. 1.] [2. 2. 2. 2.] [3. 3. 3. 3.] [4. 4. 4. 4.] [5. 5. 5. 5.] [6. 6. 6. 6.] [7. 7. 7. 7.]]

# 특정 순서의 Row 를 선택하고자 한다면 순서가 명시된 정수가 담긴 리스트를 넘긴다
# print(arr[[4, 3, 0]])  # [[4. 4. 4. 4.] [3. 3. 3. 3.] [0. 0. 0. 0.]]

# 색인으로 음수를 사용하면 끝에서부터 Row 를 선택한다
# print(arr[[-5, -7]])  # [[3. 3. 3. 3.] [1. 1. 1. 1.]]


arr = np.arange(32).reshape((8, 4))
# 다차원 색인 배열은 각각의 색인 튜플에 대응하는 1차원 배열이 선택된다
# print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
# (1, 0), (5, 3), (7, 1), (2, 2) -> [ 4 23 29 10]
# 팬시 색인의 결과는 항상 1차원이 된다


# 배열 전치와 축 바꾸기
# 배열 전치: 데이터를 복사하지 않고 데이터의 모양이 바뀐 뷰를 반환하는 기능
# ndarray 에는 transpose 메서드와 T 라는 이름의 특수한 속성을 가지고 있다
arr = np.arange(15).reshape((3, 5))
# print(arr.T)  # [[ 0  5 10] [ 1  6 11] [ 2  7 12] [ 3  8 13] [ 4  9 14]]
# print(arr.transpose())
# [[ 0  5 10] [ 1  6 11] [ 2  7 12] [ 3  8 13] [ 4  9 14]]

# 이 속성/메서드는 행렬 계산에 자주 사용된다.
# 행렬의 내적 곱
arr = np.random.randn(6, 3)
# print(np.dot(arr.T, arr))

# 다차원 배열의 경우 transpose() 메서드는 튜플로 축 번호를 받아서 치환한다
arr = np.arange(16).reshape((2, 2, 4))
# print(arr)  # [[[ 0  1  2  3] [ 4  5  6  7]] [[ 8  9 10 11] [12 13 14 15]]]

# swapaxes 메서드는 두 개의 축 번호를 받아서 배열을 뒤바꾼다
# print(arr.swapaxes(1, 2))
# [[[ 0  4] [ 1  5] [ 2  6] [ 3  7]] [[ 8 12] [ 9 13] [10 14] [11 15]]]


'''
  유니버설 함수
  ufunc라고 불리는 유니버설 함수는 ndarray 안에 있는 데이터 원소별로 연산을 수행하는 함수
  하나 이상의 스칼라값을 받아서 하나 이상의 스칼라 결괏값을 반환하는 백터화된 래퍼 함수
'''
arr = np.arange(10)

# 단항 유니버설 함수
# np.sqrt() 루트 처리
# print(np.sqrt(arr))
# [0.         1.         1.41421356 1.73205081 2.         2.23606798   2.44948974 2.64575131 2.82842712 3.        ]

# np.exp() 밑이 자연상수 e인 지수함수(e^x)로 변환
# print(np.exp(arr))
# [1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01 5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03 2.98095799e+03 8.10308393e+03]

'''
  단항 유니버설 함수
  abs, fabs : 각 원소의 점수, 부동소수점수, 복소수의 절댓값을 구한다 (복소수가 아닌 경우 빠른 연산을 위해 fabs 사용)
  sqrt : 각 원소의 제곱근을 계산 ** 0.5 와 동일
  square : 각 원소의 제곱을 계산 ** 2 와 동일
  exp : 각 원소에서 지수 e^x를 계산
  log, log10, log2, log1p : 자연소그, 로그10, 로그2, 로그(1+x)
  sign : 각 원소의 부호를 계산 -> 1은 양수, -1은 음수
  ceil : 각 원소의 소수자리를 올린다
  floor : 각 원소의 소수자리를 내린다
  rint : 각 원소의 소수자리를 반올림한다 (dtype 은 유지)
  modf : 각 원소의 몫과 나머지를 각각의 배열로 반환
  isnan : 각 원소가 숫자가 아닌지를 나타내는 불리언 배열 반환 (NaN)
  isfinite, isinf : 각각 배열의 각 원소가 유한한지 무한한지 나타내는 불리언 배열 반환
  cos, cosh, sin, sinh, tan, tanh : 일반 삼각함수와 쌍곡삼각함수
  arccos, arccosh, arcsin, arcsinh, arctan, arctanh : 역삼각함수
  logical_not : 각 원소의 논리 부정 값을 계산 ~arr 와 동일
'''

'''
  이항 유니버설 함수
  add : 두 배열에서 같은 위치의 원소끼리 더하기
  subtract : 첫 번째 배열의 원소에서 두 번째 배열 원소를 뺀다
  multiply : 배열의 원소끼리 곱하기
  divide, floor_divide : 첫 번째 배열의 원소를 두 번째 배열의 원소로 나눈다 -> floor_divide 는 몫만 취한다
  power : 첫 번째 배열의 원소를 두 번째 배열의 원소만큼 제곱한다
  maximum, fmax : 각 배열의 두 원소 중 큰 값을 반환 -> fmax는 NaN를 무시
  minimum, fmin : 각 배열의 두 원소 중 작은 값을 반환 -> fmin은 NaN을 무시
  mod : 첫 번째 배열의 원소를 두 번째 배열의 원소를 나눈 나머지를 구한다
  copysign : 첫 번째 배열의 원소의 기호를 두 번째 배열의 원소의 기호로 바꾼다
  greater, greater_equal, less, less_equal, equal, not_equal : 각각 두 원소 간의 비교 연산 결과를 불리언 배열로 반환
  logical_and, logical_or, logical_xor : 각각 두 원소 간의 논리 연산 결과 반환
'''

# 이항 유니버설 함수
x = np.random.randn(8)
y = np.random.randn(8)
# np.maximum() 은 원소별로 가장 큰 값을 계산
# print(np.maximum(x, y))

arr = np.random.randn(7) * 5
# np.modf() divmod 의 벡터화 버전으로 분수를 받아서 몫과 나머지를 함께 반환한다
remainder, whole_part = np.modf(arr)
# print(remainder)
# print(whole_part)


'''
  벡터화
  배열 연산을 사용하여 반복문을 명시적으로 제거하는 기법

  벡터화된 배열에 대한 산술 연산은 순수 파이썬 연산에 비해 2~수백 배까지 빠르다
'''

points = np.arange(-5, 5, 0.01)
# print(points)  # -5 ~ 4.99까지 0.01씩 증가하는 값들의 배열

# np.meshgrid() 두 개의 1차원 배열을 받아서 가능한 모든 (x, y) 짝을 만들 수 있는 2차원 배열 두 개를 반환
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs**2+ys**2)
# print(z)
# [[7.07106781 7.06400028 7.05693985 ... 7.04988652 7.05693985 7.06400028]
#  [7.06400028 7.05692568 7.04985815 ... 7.04279774 7.04985815 7.05692568]
#  [7.05693985 7.04985815 7.04278354 ... 7.03571603 7.04278354 7.04985815]
#  ...
#  [7.04988652 7.04279774 7.03571603 ... 7.0286414  7.03571603 7.04279774]
#  [7.05693985 7.04985815 7.04278354 ... 7.03571603 7.04278354 7.04985815]
#  [7.06400028 7.05692568 7.04985815 ... 7.04279774 7.04985815 7.05692568]]

plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
# plt.show()


# 배열 연산으로 조건절 표현
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# 순수 파이썬으로 수행 - 큰 배열을 빠르게 처리하지 못하며, 다차원 배열에선 사용할 수 없다
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
# print(result)  # [1.1, 2.2, 1.3, 1.4, 2.5]

# np.where() - 두 번째, 세 번째 인자는 배열이 아니어도 상관 없다 (스칼라값도 가능)
result = np.where(cond, xarr, yarr)

result2 = np.where(cond, xarr, -2)
# print(result2)  # [ 1.1 -2.   1.3  1.4 -2. ]


# 배열 전체 혹은 배열에서 한 축을 따르는 자료에 대한 통계를 계산하는 수학 함수
arr = np.random.randn(5, 4)

# 선택적으로 axis 인자를 받아서 해당 axis 에 대한 통계를 계산하고 한 차수 낮은 배열을 반환한다
# mean() 평균
# print(arr.mean())
# print(np.mean(arr))

# sum() 합계
# print(arr.sum())

# print(arr.mean(axis=1))  # Column 의 평균
# print(arr.sum(axis=0))  # Row 의 합

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# cumXXX() 은 중간값을 담고 있는 배열을 반환
# print(arr.cumsum())  # [ 0  1  3  6 10 15 21 28]
# print(arr.cumprod())  # [0 0 0 0 0 0 0 0]

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print(arr.cumsum(axis=0))  # [[ 0  1  2] [ 3  5  7] [ 9 12 15]]
# print(arr.cumprod(axis=1))  # [[  0   0   0] [  3  12  60] [  6  42 336]]

'''
  기본 배열 통계 메서드
  sum
  mean : 산술 평균
  std, var : 표준편자, 분산 -> 분모의 기본값은 n
  min, max : 최소, 최댓값
  argmin, argmax : 최소 원소의 색인값과 최대 원소의 색인값
  cumsum : 각 원소의 누적 합
  cumprod : 각 원소의 누적 곱
'''


# 불리언 배열을 위한 메서드
arr = np.random.randn(100)
# print((arr > 0).sum())  # 조건식이 참일 때만 실행됨

bools = np.array([False, False, True, False])
# any() 메서드는 하나 이상의 값이 True 인지 검사
# print(bools.any())  # True

# all() 메서드는 모든 원소가 True 인지 검사
# print(bools.all())  # False

# 해당 메서드는 불리언 배열이 아니어도 동작한다
# 0 이 아닌 원소는 모두 True로 간주한다 (즉 0은 False)


# 정렬
arr = np.random.randn(6)

arr.sort()
# print(arr)

# sort() 메서드는 넘긴 축을 파라미터로 받아 1차원 부분으로 정렬된다
# sort() 메서드는 배열을 직접 변경하지 않고 정렬된 결과를 가지고 있는 복사본을 반환
arr = np.random.randn(5, 3)
arr.sort(1)
# print(arr)

# 배열의 분위수를 구하는 방법은 배열을 정렬한 뒤 특정 분위의 값을 선택하는 것
large_arr = np.random.randn(1000)
large_arr.sort()
# print(large_arr[int(0.05 * len(large_arr))])  # 5% 분위수
'''
  분위수는 자료를 크기 순으로 배열할 경우 그 자료를 분할하는 역할을 하는 위치의 수치를 계산한 것이다
  분위수의 종류로는 자료를 몇 등분하는가에 따라 사분위수, 오분위수, 십분위수, 백분위수 등이 있다
'''

# 집합 관련 함수
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will'])

# np.unique() 배열 내에서 중복된 원소를 제거하고 남은 원소를 정렬한 형태로 반환
# print(np.unique(names))  # ['Bob' 'Joe' 'Will']

# np.in1d() 두 개의 배열을 인자로 받아서 첫 번째 배열의 원소가 두 번째 배열의 원소를 포함하는지 나타내는 불리언 배열 반환
values = np.array([6, 0, 0, 3, 2, 5, 6])
# print(np.in1d(values, [2, 3, 6]))
# [ True False False  True  True False  True]

'''
  배열 집합 연산
  unique(x) 배열 x에서 중복된 원소를 제거한 뒤 정렬하여 반환
  intersect1d(x, y) 배열 x와 y에 공통적으로 존재하는 원소를 정렬하여 반환
  union2d(x, y) 두 배열의 합집합 반환
  in1d(x, y) x의 원소가 y의 원소에 포함되는지 나타내는 불리언 배열 반환
  setdiff1d(x, y) x, y의 차집합 반환
  setxor1d(x, y) 한 배열에는 포함되지만 두 배열 모두에는 포함되지 않은 원소들의 집합인 대칭차집합 반환
'''


# 배열 데이터의 파일 입출력
# 디스크에서 텍스트나 바이너리 형식의 데이터를 불러오거나 저장하기
arr = np.arange(10)

# np.save() 배열 데이터를 효과적으로 디스크에 저장하고 불러오기 위한 함수
np.save('some', arr)
# 배열은 기본적으로 압축되지 않은 원시 바이너리 형식의 .npy 파일로 저장된다 (확장자가 없으면 자동적으로 확장자가 추가됨)

# np.load() 저장된 배열을 불러오는 함수 (확장자를 생략하면 안된다!)
# print(np.load('some.npy'))

# np.savez() 여러 개의 배열을 압축된 형식(.npz)으로 저장할 수 있다
# 파라미터를 키워드 인자 형태로 전달
# np.savez('arr.npz', a=arr, b=arr)

# .npz 확장자는 각각의 배열을 필요할 때 불러올 수 있도록 사전 형식의 객체에 저장
arrnpz = np.load('arr.npz')
# print(arrnpz['b'])

# 압축이 잘되는 형식의 데이터라면 numpy.savez_compressed() 를 사용해준다
# np.savez_compressed('arrnpz.npz', arr1=arr, arr2=arr2d)


'''
  선형대수
  행렬의 곱셈, 분할, 행렬식 그리고 정사각 행렬 수학을 활용하여 배열을 다룬다
  Matlab 같은 언어와 다르게 2개의 2차원 배열을 * 연산자로 곱하면 행렬 곱셈이 아니라
  대응하는 각각의 원소곱을 계산한다

  행렬 곱은 dot() 함수를 이용한다
'''
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[6, 23], [-1, 7], [8, 9]])

# np.dot(a, b) 과 a.dot(b) 는 동일한 결과를 리턴한다
# print(x.dot(y))  # 행렬 곱
# print(np.dot(x, y))  # 행렬 곱

# 2차원 배열과 곱셈이 가능한 크기의 1차원 배열 간 행렬 곱셈의 결과는 1차원 배열이다
# print(np.dot(x, np.ones(3)))  # [ 6. 15.]
# @를 사용하여 행렬 곱을 수행할 수 있다
# print(x @ np.ones(3))  # [ 6. 15.]

# numpy.linalg는 행렬의 분할과 역행렬, 행렬식 등을 포함하고 있다
X = np.random.randn(5, 5)

# T 속성은 전치행렬을 의미
mat = X.T.dot(X)

# inv() 역행렬을 의미
# print(inv(mat))

# print(mat.dot(inv(mat)))

# qr() 은 QR분해를 계산
q, r = qr(mat)
# print(r)

'''
  numpy.linalg 함수
  diag : 정사각 행렬의 대각/비대각 원소를 1차원 배열로 반환하거나 1차원 배열을 대각선 원소로 하고 나머지는 0으로 채운 단위행렬 반환
  dot : 행렬 곱
  trace : 행렬의 대각선 원소의 합을 계산
  det : 행렬식을 계산
  eig : 정사각 행렬의 고윳값과 고유벡터를 계산
  inv : 정사각 행렬의 역행렬 계산
  pinv : 정사각 행렬의 무어-펜로즈 유사역원 역행렬 계산
  qr : QR 분해 계산
  svd : 특잇값 분해(SVD) 계산
  solve : A 가 정사각 행렬일 때 Ax = b 를 만족하는 x
  lstsq : Ax = b 를 만족하는 최소제곱해
'''


# 난수 생성
# numpy.random 모듈은 다양한 종류의 확률분포로부터 효과적으로 표본값을 생성하는데 주로 사용
# normal() 은 표준정규분포로부터 4x4 크기의 표본 생성
samples = np.random.normal(size=(4, 4))
# print(samples)

# 파이썬 내장 random 모듈은 한번에 하나의 값만 생성 가능 numpy.random 은 표본을 생성하는데 파이썬 내장 모듈보다 수십 배 이상 빠름
N = 100000
samples = [normalvariate(0, 1) for _ in range(N)]
np.random.normal(size=N)
# 유사 난수 : 난수 생성기의 시드값에 따라 정해진 난수를 알고리즘으로 생성
# 난수 생성기의 시드값은 np.random.seed 를 이용하여 변경 가능
np.random.seed(1234)

# np.random.RandomState() 다른 난수 생성기로부터 격리된 난수 생성기를 만들 수 있음
rng = np.random.RandomState(1234)
rng.randn(10)
print(rng)

'''
  numpy.random 함수
  seed : 난수 생성기 시드를 지정
  permutation : 순서를 임의로 바꾸거나 임의의 순열을 반환
  shuffle : 리스트나 배열의 순서를 뒤섞기
  rand : 균등분포에서 표본을 추출
  randint : 주어진 최소/최대 범위 안에서 임의의 난수를 추출
  randn : 표준 편차가 1이고 평균값이 0인 정규분포에서 표본을 추출
  binomaial : 이항분포에서 표본을 추출
  normal : 정규분포(가우시안)에서 표본을 추출
  beta : 베타분포에서 표본을 추출
  chisquare : 카이제곱분포에서 표본을 추출
  gamma : 감마분포에서 표본을 추출
  uniform : 균등분포에서 표본을 추출
'''
