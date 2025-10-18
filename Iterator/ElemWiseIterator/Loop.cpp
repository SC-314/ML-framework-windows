#include "ElemWiseIterator.h"


Tensor loop(ElemWiseIterator iter) {
    int totalElements = 1;
    for (size_t i = 0; i < iter.resultShape.size(); i++) {
        totalElements *= iter.resultShape[i];
    }

    std::vector<double>result({});

    for (int n = 0; n < totalElements; n++) {
        int index = n;

        size_t elemA = 0;
        size_t elemB = 0;

        for (size_t idx = 0; idx < iter.resultShape.size(); idx++){
            int dimIdx = index / iter.operationStrides[2][idx];
            index = index % iter.operationStrides[2][idx];

            elemA += (dimIdx * iter.operationStrides[0][idx]);
            elemB += (dimIdx * iter.operationStrides[1][idx]);
        }
        double value = iter.op((*iter.inputs[0].data)[elemA],(*iter.inputs[1].data)[elemB]);
        result.push_back(value);
    }

    return Tensor(result, iter.resultShape, iter.operationStrides[2]);
}