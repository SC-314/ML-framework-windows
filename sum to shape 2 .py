xMatrix = [[1,2],
           [3,4]]
yMatrix = [[0,0]]


x = [1,2,3,4]
xShape = [2,2]
xStride = [2,1]

y = [0,0]
yShape = [2,1]
yStride = [1,1]

input_indices = [0,0]   # 0,0: input=0, output=0
                        # 0,1: input=1, output=1
                        # 1,0: input=2, output=0
                        # 1,1: input=3, output=1
for j in range(4):
    input_flat = 0
    output_flat = 0
    # print(input_indices)

    for i in range(2):
        input_flat += xStride[i] * input_indices[i]
        if (yShape[i] == 1):
            output_flat += 0
        else:
            output_flat += yStride[i] * input_indices[i]
    
    # print(input_flat)
    # print(output_flat)
    
    y[output_flat] += x[input_flat]

    for i in range(len(input_indices))[::-1]:

        input_indices[i] += 1

        if input_indices[i] == xShape[i]:
            input_indices[i] = 0
        else:
            break

print(y)

# [0,0]

# [0,1]

# [0,2]
# [1,0]