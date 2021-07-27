import numpy as np
score = [20, 15, 26, 32, 18, 28, 35, 14, 26, 22, 17]
mean =  sum(score) / len(score)
print("The mean: ",round(mean,2))

sd = np.std(score)
print("The standard deviation: ",round(sd,2))

sc=[]
for i in score:
    z= (i-mean)/sd
    sc.append(z)

print("The standard scores: ")
for i in sc:
    print(round(i,2), end=" ")

print("\nThe student scores",end=" ")
count=0
for i in range(0,len(sc)):
    if sc[i]<-1 and count==0:
        print(score[i], end=" ")
        count+=1
    elif sc[i]<-1 and count==1:
        print("and",score[i], end=" ")

print("stand out to receive an F")
