#pragma once
#include "../Tensor/Tensor.h"
#include <cstddef>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor operator()(Tensor& input) = 0;
    virtual std::vector<Tensor*> get_parameters() = 0;
    
    Tensor weights;
    Tensor bias;
};

class Linear : public Layer {
public:
    Tensor weights;
    Tensor bias;
    Tensor A;
    Linear() {
        weights = Tensor();
        bias = Tensor();
    }
    Linear(size_t input_layers, size_t output_layers);

    std::vector<Tensor*> get_parameters();

    Tensor operator()(Tensor& input);
};