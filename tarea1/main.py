m_1 = [
    [1,2,3],
    [2,3,4],
    [1,1,1]
]
m_2 = [
    [4,5,6],
    [7,8,9],
    [4,5,7]
]

res = [[0 for _ in range(3)] for _ in range(3)]

# wanted to make it inline but its too complex
# for some reason on these problems numpy fails
for i in range(3):
    for j in range(3):
        res[i][j] = sum([m_1[i][k] * m_2[k][j] for k in range(3)])

res_flat = list(map(str,res[0] + res[1] + res[2]))
print("\n".join(res_flat))






