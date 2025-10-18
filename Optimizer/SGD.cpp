#include "./Optimizer.h"
#include <functional>


Optim::SGD::SGD(double learning_rate, std::vector<Tensor*> param_ptrs)
    : learning_rate(learning_rate), param_ptrs_(param_ptrs) {}

void Optim::SGD::zero_grads() {
    for (Tensor* tensor : param_ptrs_) {
        for (size_t i = 0; i < tensor->grad->size(); i++) {
            (*tensor->grad)[i] = 0.0;
        }
    }
}

void Optim::SGD::apply_grads() {
    for (Tensor* tensor : param_ptrs_) {
        for (size_t i = 0; i < tensor->data->size(); i++) {
            (*tensor->data)[i] = (*tensor->data)[i] - (learning_rate * (*tensor->grad)[i]);
        }
    }
}