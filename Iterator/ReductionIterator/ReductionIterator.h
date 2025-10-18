#pragma once

#include "../../Tensor/Tensor.h"
#include <functional>
#include <string.h>

class ReductionIterator;

Tensor loop(ReductionIterator iter);

class ReductionIterator {
private:
    friend class ReductionIteratorConfig;
    friend class Tensor loop(ReductionIterator iter);

    
    std::function<double(std::vector<double>)> op;
    std::vector<size_t> resultShape;
    std::vector<Tensor> inputs;
    std::vector<std::vector<size_t>> operationStrides;

    explicit ReductionIterator(std::vector<Tensor> inputs, std::function<double(std::vector<double>)> op);

public:
    class ReductionIteratorConfig {
    private:
        std::function<double(std::vector<double>)> op;
        std::vector<Tensor> inputs = {};

    public:
        explicit ReductionIteratorConfig(std::function<double(std::vector<double>)> op);
        explicit ReductionIteratorConfig(const std::string op);
        
        ReductionIteratorConfig& add_input(const Tensor& tensor);

        ReductionIterator build();
        
    };

    Tensor for_each();
};