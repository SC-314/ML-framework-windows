xMatrix = [[1,2],
           [3,4]]
yMatrix = [[0,0]]


x = [1,2,3,4]
xShape = [2,2]
xStride = [2,1]

y = [0,0]
yShape = [1,2]
yStride = [2,1]


input_indices = [0,1] # 1,0 should give output of 0


input_flat = 0
for i in range(2):
    input_flat += xStride[i] * input_indices[i]
print(input_flat)


output_flat = 0
for i in range(2):
    if (yShape[i] == 1):
        output_flat += 0
    else:
        output_flat += yStride[i] * input_indices[i]
print(output_flat)