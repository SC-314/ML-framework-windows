#pragma once
#include <memory>
#include <vector>

class Tensor;

class GraphNode {
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_; // here
    virtual void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) = 0;
    virtual void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) = 0; // here
};

