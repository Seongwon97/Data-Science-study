from math import sqrt,ceil
import sys
def isPrime(num):
    for i in range(2,ceil(sqrt(num))+1):
        if num%i==0:
            return 0
        else:
            continue
    return 1

n=int(input("Enter the number: "))
if not 2<n<32767:
    sys.exit(0)

result=isPrime(n)
if result == 1:
    print(n, "is prime number.")
else:
    print(n, "is not prime number")
