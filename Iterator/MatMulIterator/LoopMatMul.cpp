#include "MatMulIterator.h"
#include <cstddef>
#include <vector>
#include <algorithm>

Tensor loopMatMul(MatMulIterator iter) {
    int totalElements = 1;
    for (size_t i = 0; i < iter.resultShape.size(); i++) {
        totalElements *= iter.resultShape[i];
    }

    std::vector<double>result({});

    size_t total = 1;
    for (size_t i = 0; i < iter.resultShape.size() - 2; i++) { // -3 so we exclude the final two
        total *= iter.resultShape[i];
    }

    for (int n = 0; n < total; n++) {

        int index;
        if (iter.operationStrides[2].size() > 0) {
            index = n * iter.operationStrides[2][iter.operationStrides[2].size() - 1];
        }
        else {
            index = n * 1;
        }

        size_t elemA = 0;
        size_t elemB = 0;

        for (size_t idx = 0; idx < iter.operationStrides[2].size(); idx++){
            int dimIdx = index / iter.operationStrides[2][idx];
            index = index % iter.operationStrides[2][idx];

            elemA += (dimIdx * iter.operationStrides[0][idx]);
            elemB += (dimIdx * iter.operationStrides[1][idx]);
        }

        std::vector<double> hello = *iter.inputs[0].data;

        for (size_t i = 0; i < iter.resultShape[iter.resultShape.size() - 2]; i++) {
            for (size_t j = 0; j < iter.resultShape[iter.resultShape.size() - 1]; j++) {
                double resultElem = 0;
                for (size_t k = 0; k < iter.inputs[0].shape[iter.inputs[0].shape.size() - 1]; k++) {
                    size_t rowStride = (iter.inputs[0].shape)[iter.inputs[0].shape.size() - 1];
                    size_t colStride = (iter.inputs[1].shape)[iter.inputs[1].shape.size() - 1];
                    resultElem += (
                        (*iter.inputs[0].data)[elemA + i * rowStride + k]
                        * (*iter.inputs[1].data)[elemB + j + k*colStride]
                    );
                }
                result.push_back(resultElem);
            }
        }
    }
    
    size_t stride = 1;
    std::vector<size_t>resultStrides({});
    std::vector<size_t> strideShape = iter.resultShape;
    std::reverse(strideShape.begin(), strideShape.end());


    for (auto a : strideShape) {
        
        resultStrides.push_back(stride);
        stride *= a;
    }

    std::reverse(resultStrides.begin(), resultStrides.end());

    return Tensor(result, iter.resultShape, resultStrides);
}