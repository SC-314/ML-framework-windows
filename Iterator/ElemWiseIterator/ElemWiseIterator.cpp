#include "./ElemWiseIterator.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cstdlib>

ElemWiseIterator::ElemWiseIterator(
    std::vector<Tensor> inputs,
    std::vector<size_t> resultShape,
    std::vector<std::vector<size_t>> operationStrides,
    std::function<double(double, double)> op):
    inputs(inputs), resultShape(resultShape), operationStrides(operationStrides), op(op) {}

ElemWiseIterator::ElemWiseIteratorConfig::ElemWiseIteratorConfig(std::function<double(double, double)> op): op(op) {};

ElemWiseIterator::ElemWiseIteratorConfig& ElemWiseIterator::ElemWiseIteratorConfig::add_input(const Tensor& tensor) {
    inputs.push_back(tensor);
    return *this;
}

ElemWiseIterator ElemWiseIterator::ElemWiseIteratorConfig::build() {
    std::vector<size_t> aShape = inputs[0].shape;
    std::vector<size_t> bShape = inputs[1].shape;
    std::vector<size_t>resultShape({});
    std::vector<size_t>aBroadcastStrides({});
    std::vector<size_t>bBroadcastStrides({});
    std::vector<size_t>resultStrides({});
    size_t aBroadcastDim = 1;
    size_t bBroadcastDim = 1;
    size_t resultBroadcastDim = 1;

    std::reverse(aShape.begin(), aShape.end());
    std::reverse(bShape.begin(), bShape.end());

    size_t newSize = std::max(aShape.size(), bShape.size());

    aShape.resize(newSize, 1);
    bShape.resize(newSize, 1);

    for (size_t i = 0; i < newSize; i++) {
        if (aShape[i] == bShape[i]) {
            aBroadcastStrides.push_back(aBroadcastDim);
            bBroadcastStrides.push_back(bBroadcastDim);
            resultStrides.push_back(resultBroadcastDim);
            resultShape.push_back(aShape[i]);
            resultBroadcastDim *= aShape[i];
        } else if (aShape[i] == 1) {
            aBroadcastStrides.push_back(0);
            bBroadcastStrides.push_back(bBroadcastDim);
            resultStrides.push_back(resultBroadcastDim);
            resultShape.push_back(bShape[i]);
            resultBroadcastDim *= bShape[i];
        } else if (bShape[i] == 1) {
            aBroadcastStrides.push_back(aBroadcastDim);
            bBroadcastStrides.push_back(0);
            resultStrides.push_back(resultBroadcastDim);
            resultShape.push_back(aShape[i]);
            resultBroadcastDim *= aShape[i];
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
    
    std::vector<std::vector<size_t>>operationStrides({aBroadcastStrides, bBroadcastStrides, resultStrides});
    return ElemWiseIterator(inputs, resultShape, operationStrides, op);
}

Tensor ElemWiseIterator::for_each() {
    return loop(*this);
}