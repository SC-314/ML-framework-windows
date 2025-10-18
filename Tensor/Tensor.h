#pragma once
#include <iostream>
#include <ostream>
#include <vector>
#include <memory>

#include "../AutoGradEngine/GraphNode.h"

class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::shared_ptr<std::vector<double>> data;
    std::shared_ptr<std::vector<double>> grad;
    std::shared_ptr<GraphNode> grad_fn;
    bool final = false;
    bool requires_grad = true;

    Tensor(std::vector<double> data, std::vector<size_t> shape, std::vector<size_t> strides);
    Tensor();

    Tensor operator=(const Tensor& other);
    Tensor copy();
    Tensor operator+(Tensor& other);
    Tensor operator-(Tensor& other);
    Tensor operator*(Tensor& other);
    Tensor operator/(Tensor& other);
    Tensor operator&(Tensor& other);
    
    void backward(bool final = false);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);

    void set_grad_fn(std::shared_ptr<GraphNode> node);
};

#include "../BinaryOps/BinaryOps.h"
#include "../MatMulOp/MatMul.h"
