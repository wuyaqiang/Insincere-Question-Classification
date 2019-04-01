x = int(input())
y = int(input())
z = int(input())

if x == y == z:
	print("等边三角形")
elif x==y or y==z or z==x:
	print("等腰三角形")
else:
	print("既不是等边，也不是等腰")