xMatrix = [[1,2],
           [3,4]]
yMatrix = [[0,0]]


x = [1,2,3,4]
xShape = [2,2]
xStride = [2,1]

y = [0,0]
yShape = [1,2]
yStride = [2,1]

input_indices = [0,0] # 1,0 should give output of 0


input_flat = 0
output_flat = 0

for j in range(3):

    for i in range(2):
        input_flat += xStride[i] * input_indices[i]
        if (yShape[i] == 1):
            output_flat += 0
        else:
            output_flat += yStride[i] * input_indices[i]
        
    y[output_flat] += x[input_flat]

    for i in range(len(input_indices))[::-1]:
        input_indices[i] += 1

        if input_indices[i] == xShape[i]:
            input_indices[i] = 0
        else:
            break
    print(input_indices)

print("\n")
print(y)