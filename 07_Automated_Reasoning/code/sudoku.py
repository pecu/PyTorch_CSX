from z3 import *
import time

N =9
su = [[z3.Int('X_%d_%d' %(i,j)) for j in range(9)] for i in range(9)]
s=Solver()
	

for i in range(N):
	for j in range(N):
		s.add(su[i][j]>0,su[i][j]<10)


for k in range(N):
    for i in range(N):
        for j in range(i +1,N):
            s.add(su[k][i]!= su[k][j])
            s.add(su[i][k]!= su[j][k])

for k in range(0,7,3):	
	for l in range(0,7,3):
		for i in range(N):
			for j in range(i+1,N):
				s.add(su[i/3+k][i%3+l]!=su[j/3+k][j%3+l])


s.add(su[0][1]==2)
s.add(su[1][3]==6)
s.add(su[1][8]==3)
s.add(su[2][1]==7)
s.add(su[2][2]==4)
s.add(su[2][4]==8)
s.add(su[3][5]==3)
s.add(su[3][8]==2)
s.add(su[4][1]==8)
s.add(su[4][4]==4)
s.add(su[4][7]==1)
s.add(su[5][0]==6)
s.add(su[5][3]==5)
s.add(su[6][4]==1)
s.add(su[6][6]==7)
s.add(su[6][7]==8)
s.add(su[7][0]==5)
s.add(su[7][5]==9)
s.add(su[8][7]==4)	

if s.check()==sat:
	print  s.model()