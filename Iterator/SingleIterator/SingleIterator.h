#pragma once

#include "../../Tensor/Tensor.h"
#include <functional>
#include <string.h>

class SingleIterator;

Tensor loop(SingleIterator iter);

class SingleIterator {
private:
    friend class SingleIteratorConfig;
    friend class Tensor loop(SingleIterator iter);

    
    std::function<double(double)> op;
    std::vector<size_t> resultShape;
    std::vector<Tensor> inputs;
    std::vector<std::vector<size_t>> operationStrides;

    explicit SingleIterator(std::vector<Tensor> inputs, std::function<double(double)> op);

public:
    class SingleIteratorConfig {
    private:
        std::function<double(double)> op;
        std::vector<Tensor> inputs = {};

    public:
        explicit SingleIteratorConfig(const std::function<double(double)> op);
        explicit SingleIteratorConfig(const std::string op);
        
        SingleIteratorConfig& add_input(const Tensor& tensor);

        SingleIterator build();
        
    };

    Tensor for_each();
};