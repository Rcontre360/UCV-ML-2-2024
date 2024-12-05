res = 0
for i in range(1,7):
    for j in range(1,7):
        if i != j and i + j == 6:
            res+=1

print(res)
