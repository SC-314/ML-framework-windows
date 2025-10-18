#include "./MatMulIterator.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cstdlib>

MatMulIterator::MatMulIterator(
    std::vector<Tensor> inputs,
    std::vector<size_t> resultShape,
    std::vector<std::vector<size_t>> operationStrides,
    std::function<double(double, double)> op):
    inputs(inputs), resultShape(resultShape), operationStrides(operationStrides), op(op) {}

MatMulIterator::MatMulIteratorConfig::MatMulIteratorConfig(std::function<double(double, double)> op): op(op) {};

MatMulIterator::MatMulIteratorConfig& MatMulIterator::MatMulIteratorConfig::add_input(const Tensor& tensor) {
    inputs.push_back(tensor);
    return *this;
}

MatMulIterator MatMulIterator::MatMulIteratorConfig::build() {
    std::vector<size_t> aShape = inputs[0].shape; // (1,4,5)
    std::vector<size_t> bShape = inputs[1].shape; // (2,3,5,3)
    std::vector<size_t>resultShape({});
    std::vector<size_t>aBroadcastStrides({});
    std::vector<size_t>bBroadcastStrides({});
    std::vector<size_t>resultStrides({});
    size_t aBroadcastDim = 1;
    size_t bBroadcastDim = 1;
    size_t resultBroadcastDim = 1;

    std::reverse(aShape.begin(), aShape.end()); // (5,4,1)
    std::reverse(bShape.begin(), bShape.end()); // (3,5,3,2)

    size_t newSize = std::max(aShape.size(), bShape.size()); // 4

    aShape.resize(newSize, 1); // (5,4,1,1)
    bShape.resize(newSize, 1); // (3,5,3,2)

    if (aShape[0] == bShape[1]) { // yes both are 5
        aBroadcastDim *= (aShape[0] * aShape[1]);
        bBroadcastDim *= (bShape[0] * bShape[1]);
        // aBroadcastStrides.push_back(aBroadcastDim);
        // bBroadcastStrides.push_back(bBroadcastDim);
        resultShape.push_back(bShape[0]); // (3)
        resultShape.push_back(aShape[1]);// (3,4)
        resultBroadcastDim  *= (aShape[1] * bShape[0]);
        // resultStrides.push_back(resultBroadcastDim);
    } else {
        std::cerr << "Error: Tensors are not compatible" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 2; i < newSize; i++) {
        if (aShape[i] == bShape[i]) {
            aBroadcastStrides.push_back(aBroadcastDim);
            bBroadcastStrides.push_back(bBroadcastDim);
            resultStrides.push_back(resultBroadcastDim);
            resultShape.push_back(aShape[i]);
            resultBroadcastDim *= aShape[i];
        } else if (aShape[i] == 1) {
            resultStrides.push_back(resultBroadcastDim);
            resultBroadcastDim *= bShape[i];
            aBroadcastStrides.push_back(0);
            bBroadcastStrides.push_back(bBroadcastDim);
            resultStrides.push_back(resultBroadcastDim);
            resultShape.push_back(bShape[i]);
            
        } else if (bShape[i] == 1) {
            resultStrides.push_back(resultBroadcastDim);
            resultBroadcastDim *= aShape[i];
            aBroadcastStrides.push_back(aBroadcastDim);
            bBroadcastStrides.push_back(0);
            resultShape.push_back(aShape[i]);
            
        } else {
            std::cerr << "Error: Tensors are not compatible" << std::endl;
            exit(EXIT_FAILURE);
        }
        aBroadcastDim *= aShape[i];
        bBroadcastDim *= bShape[i];
    }
    
    std::reverse(resultShape.begin(), resultShape.end());
    std::reverse(aBroadcastStrides.begin(), aBroadcastStrides.end());
    std::reverse(bBroadcastStrides.begin(), bBroadcastStrides.end());
    std::reverse(resultStrides.begin(), resultStrides.end());

    if (aBroadcastStrides.empty()) {
        aBroadcastStrides = {};
    }
    if (bBroadcastStrides.empty()) {
        bBroadcastStrides = {};
    }
    if (resultStrides.empty()) {
        resultStrides = {};
    }
    
    std::vector<std::vector<size_t>> operationStrides = {};
    operationStrides.push_back(aBroadcastStrides);
    operationStrides.push_back(bBroadcastStrides);
    operationStrides.push_back(resultStrides);
    return MatMulIterator(inputs, resultShape, operationStrides, op);
}

Tensor MatMulIterator::for_each() {
    return loopMatMul(*this);
}