#pragma once

#include "../../Tensor/Tensor.h"
#include <functional>
#include <string.h>

class ElemWiseIterator;

Tensor loop(ElemWiseIterator iter);

class ElemWiseIterator {
private:
    friend class ElemWiseIteratorConfig;
    friend class Tensor loop(ElemWiseIterator iter);

    
    std::function<double(double, double)> op;
    std::vector<size_t> resultShape;
    std::vector<Tensor> inputs;
    std::vector<std::vector<size_t>> operationStrides;

    explicit ElemWiseIterator(std::vector<Tensor> inputs, std::vector<size_t> resultShape, std::vector<std::vector<size_t>> operationStrides, std::function<double(double, double)> op);

public:
    class ElemWiseIteratorConfig {
    private:
        std::function<double(double, double)> op;
        std::vector<Tensor> inputs = {};

    public:
        explicit ElemWiseIteratorConfig(const std::function<double(double, double)> op);
        explicit ElemWiseIteratorConfig(const std::string op);
        
        ElemWiseIteratorConfig& add_input(const Tensor& tensor);

        ElemWiseIterator build();
        
    };

    Tensor for_each();
};