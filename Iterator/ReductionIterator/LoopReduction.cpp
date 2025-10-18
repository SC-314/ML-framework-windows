#include "ReductionIterator.h"
#include<vector>

Tensor loop(ReductionIterator iter) {
    
    int totalElements = 1;
    for (size_t i = 0; i < iter.inputs[0].shape.size(); i++) {
        totalElements *= iter.inputs[0].shape[i];
    }

    // std::cout << "total elements: " << totalElements << std::endl;

    double a = iter.op(*iter.inputs[0].data);
    
    std::vector<double> result = std::vector({a});
    
    // std::cout << "CP1" << std::endl;

    // for (auto i : result) {
    //     std::cout << i << ", ";
    // }

    return Tensor(result, std::vector<size_t>({1}), std::vector<size_t>({1}));
}