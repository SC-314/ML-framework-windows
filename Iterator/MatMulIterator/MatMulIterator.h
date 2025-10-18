#pragma once

#include "../../Tensor/Tensor.h"
#include <functional>
#include <string.h>

class MatMulIterator;

Tensor loopMatMul(MatMulIterator iter);

class MatMulIterator {
private:
    friend class MatMulIteratorConfig;
    friend class Tensor loopMatMul(MatMulIterator iter);

    
    std::function<double(double, double)> op;
    std::vector<size_t> resultShape;
    std::vector<Tensor> inputs;
    std::vector<std::vector<size_t>> operationStrides;

    explicit MatMulIterator(std::vector<Tensor> inputs, std::vector<size_t> resultShape, std::vector<std::vector<size_t>> operationStrides, std::function<double(double, double)> op);

public:
    class MatMulIteratorConfig {
    private:
        std::function<double(double, double)> op;
        std::vector<Tensor> inputs;

    public:
        explicit MatMulIteratorConfig(const std::function<double(double, double)> op);
        explicit MatMulIteratorConfig(const std::string op);
        
        MatMulIteratorConfig& add_input(const Tensor& tensor);

        MatMulIterator build();
        
    };

    Tensor for_each();
};

