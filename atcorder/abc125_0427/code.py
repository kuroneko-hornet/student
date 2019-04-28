
# f = open('input.txt')
# args = f.readlines()
# args = [i.strip('\n') for i in args]


# a = int(args[0])
# b, c = [int(i) for i in args[1].split()]
# s = int(args[2])


# print(f'{a+b+c} {s}')


# A
'''
a, b, t = map(int, input().split())
print(b*(t//a))
'''

# B
# n = int(input())
# v = [int(i) for i in input().split()]
# c = [int(i) for i in input().split()]
# r = 0
# for i,val in enumerate(v):
#     if val > c[i]:
#         r += val - c[i]
# print(r)

# C
from time import time

def fermat(n):
    if n % 2 == 0 or n == 1:
        return 0 
    if pow(2, n-1, n) == 1:
        return 1


def gcd(a,b):
    if b==0:
        return a
    return gcd(b,a%b)


def get_gcd(x,n):
    element = gcd(x[0], x[1])
    for i in range(2,n):
        element = gcd(element,x[i])
    return element


n = int(input())
a = [int(i) for i in input().split()]
prime_n = 0


if n < 3:
    exit(print(max(a)))
interim_max = 0
for i in range(n-1):
    prime_n += fermat(a[i])
    if prime_n > 2:
        exit(print(1))
    max_gcd = get_gcd(a[:i]+a[i+1:], n-1)
    if interim_max < max_gcd:
        interim_max = max_gcd
print(interim_max)

        # index = i
# a[index] = interim_max

# m = float('inf')
# for i in range(n-1):
#     diff = sum([abs(a[i]-j) for j in a])
#     if m > diff:
#         index = i
#         m = diff
# if False: #変えない時
#     print()
# else:
#     a[index] = a[index+1]
# print(a)
# print(reduce(math.gcd, a))




# D
