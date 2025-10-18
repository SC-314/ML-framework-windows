#include "./Layer.h"
#include <cstdlib>
#include <ctime>
#include <memory>

Linear::Linear(size_t input_layers, size_t output_layers) {
    std::vector<double>weightsData({});
    for (size_t i = 0; i < (input_layers * output_layers); i++) {
        weightsData.push_back(static_cast<float>(std::rand()) / RAND_MAX - 1);
   }

    std::vector<double>biasData({});
    for (size_t i = 0; i < (output_layers); i++) {
        biasData.push_back(static_cast<float>(std::rand()) / RAND_MAX - 1);
   }

   weights = Tensor(weightsData,
    std::vector<size_t>({input_layers, output_layers}),
    std::vector<size_t>({output_layers, 1}));

   bias = Tensor(biasData,
    std::vector<size_t>({1, output_layers}),
    std::vector<size_t>({output_layers, 1}));

    
    weights.grad = std::make_shared<std::vector<double>>(std::vector<double>(input_layers * output_layers, 0.0));
    bias.grad = std::make_shared<std::vector<double>>(std::vector<double>(output_layers, 0.0));
}

std::vector<Tensor*> Linear::get_parameters() {
    return {&weights, &bias};
}

Tensor Linear::operator()(Tensor& input) {
    Tensor C = input & weights;
    this->A = C;
    this->A.data = C.data; // it didmt fix the data notbinmg in iter
    this->A.grad_fn = C.grad_fn;
    this->A.grad = C.grad;
    // this->A.requires_grad = false;
    Tensor B = A + bias;
    return B;
}
