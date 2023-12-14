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

print(arr1, '\n', arr2)
print(arr2.ndim)  # 2
print(arr1.dtype)  # float64

# zeros / ones
print(np.zeros(10))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.zeros((2, 3)))  # [[0. 0. 0.] [0. 0. 0.]]
print(np.ones(5))  # [1. 1. 1. 1. 1.]

# empty -> 0으로 초기화된 배열을 반환하지 않고 가비지 값으로 채워진 배열을 반환한다
print(np.empty(10))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

#

# asarray: 입력 데이터를 ndarray로 변환하지만 입력 데이터가 이미 ndarray 일 경우 복사가 일어나진 않는다

# arange: 내장 range 함수와 유사하지만 리스트 대신 ndarray를 반환
print(np.arange(15))  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
