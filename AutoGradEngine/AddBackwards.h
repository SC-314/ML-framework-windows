#pragma once
#include "GraphNode.h"
#include "../Tensor/Tensor.h"
#include <functional>

class AddBackward : public GraphNode {
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_; // here
    void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) override;
    void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) override; // here
};


class SubBackward : public GraphNode {
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_; // here
    void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) override;
    void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) override; // here
};

class MulBackward : public GraphNode {
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_; // here
    void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) override;
    void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) override; // here
};

class DivBackward : public GraphNode {
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_; // here
    void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) override;
    void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) override; // here
};

class MatMulBackward : public GraphNode {
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_; // here
    void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) override;
    void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) override; // here
};
