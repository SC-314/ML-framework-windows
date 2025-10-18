#include "./AddBackwards.h"

#include <memory>
#include <numeric>
#include <functional>

#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>



void calc_gradient(Tensor& gradTensor,
std::vector<std::reference_wrapper<Tensor>>& save_tensors_) {

    Tensor& A = save_tensors_[0];

    std::vector<double> gradA = {};

    for (size_t i = 0; i < (*A.data).size(); i++) {
        if ((*A.data)[i] > 0) {
            gradA.push_back((*gradTensor.data)[i]);
        } else {
            gradA.push_back(0);
        }
        
    }

    for (size_t i = 0; i < A.data->size(); i++) {
        (*A.grad)[i] = (*A.grad)[i] + gradA[i];
    }
}


void ReLUBackward::backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) { // grad_output is the gradient inputted into the function
    // std::cout << "MUL HAS BEEN CALLED" << std::endl;

    Tensor grad_tensor = Tensor(*grad_output, shape, stride);

    calc_gradient(
        grad_tensor,
        save_tensors_);
    
    for (Tensor& A : save_tensors_) {
        A.backward(false);
    }
}

void ReLUBackward::save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors) { //
    save_tensors_ = save_tensors;
};