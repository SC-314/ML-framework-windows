A = [[[1,2],
      [3,4],
      [5,6]],
      [7,8],
      [9,10],
      [11,12]]

B = [[[1,2,3,4],
      [5,6,7,8]]]

a = [1,2,3,4,5,6,7,8,9,10,11,12]
b = [1,2,3,4,5,6,7,8]
c = []

aShape = (2,3,2)
bShape = (1,2,4)
cShape = (2,3,4)

aStrides = [6]
bStrides = [0]
cStrides = [12]

aMatrixStrides = (2,1)
bMatrixStrides = (2,1)
cMatrixStrides = (4,1)

for n in range(2):
    elem_a = 0
    elem_b = 0
    index = n * cStrides[-1]

    for idx in range(len(cStrides)):
        dim_idx = index // cStrides[idx]
        index = index % cStrides[idx]

        elem_a += (dim_idx * aStrides[idx])
        elem_b += (dim_idx * bStrides[idx])


    for i in range(3):
        for j in range(4):
            result = 0
            for k in range(2):
                result += (a[elem_a + i *2 + k] * b[elem_b + j + k*4])
            c.append(result)
print(c)