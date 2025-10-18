#include "SingleIterator.h"


Tensor loop(SingleIterator iter) {
    
    int totalElements = 1;
    for (size_t i = 0; i < iter.inputs[0].shape.size(); i++) {
        totalElements *= iter.inputs[0].shape[i];
    }

    // std::cout << "total elements: " << totalElements << std::endl;
    
    std::vector<double>result({});

    for (int n = 0; n < totalElements; n++) {
        double value = iter.op((*iter.inputs[0].data)[n]);
        result.push_back(value);
    }
    
    // std::cout << "CP1" << std::endl;

    // for (auto i : result) {
    //     std::cout << i << ", ";
    // }

    return Tensor(result, iter.inputs[0].shape, iter.inputs[0].strides);
}