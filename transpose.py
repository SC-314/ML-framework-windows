x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
y = [0] * len(x)
xStrides = [12, 4, 1]
yStides = [12, 3, 1]

for i in range(len(x)):
    index = i
    element = 0
    opStrides = []
    normalStrides = []

    for stride in xStrides[:-2]:
        dimIndex = index // stride
        index = index % stride
        normalStrides.append(dimIndex)

    for stride in xStrides[-2:]:
        dimIndex = index // stride
        index = index % stride
        opStrides.append(dimIndex)
    print(normalStrides)
    print(opStrides)
    

    opStrides = normalStrides + opStrides[::-1]

    print("SPDOIFJPSOJi")
    for a,b in zip(yStides, opStrides):
        element += a*b
    
    y[element] = x[i]

print(y)