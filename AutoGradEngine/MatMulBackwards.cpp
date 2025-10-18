#include "./AddBackwards.h"

#include <memory>
#include <numeric>
#include <functional>

#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>


std::vector<double> sum_to_shape(std::vector<double> input, std::vector<size_t>& target_shape, std::vector<size_t> shape, std::vector<size_t> stride) {
    // Calculate output size and strides
    size_t output_size = std::accumulate(target_shape.begin(), target_shape.end(), 
                                       size_t(1), std::multiplies<>()); // calculate the number of elements in the output shape
    
    std::vector<size_t> output_strides(target_shape.size());
    output_strides.back() = 1;
    for (int i = static_cast<int>(target_shape.size()) - 2; i >= 0; i--) {
        output_strides[i] = output_strides[i + 1] * target_shape[i + 1];
    } // target shape = (2,3,4) -> output strides (12,4,1)
    
    std::vector<double> result(output_size, 0.0);
    
    // Iterate through all input elements
    std::vector<size_t> input_indices(shape.size(), 0);
    
    do {
        // Calculate the current input_flat of the currnet input_indices starting at [0,...,0,0]
        // so we can use this to iterate through the input_indices
        size_t input_flat = 0;
        for (size_t i = 0; i < shape.size(); i++) { // takes the input_indices and converts to the flat index
            input_flat += input_indices[i] * stride[i];
        }
        
        // Calculate the current output index of output_idx, 
        size_t output_flat = 0;
        for (size_t i = 0; i < target_shape.size(); i++) {
            size_t output_idx = (target_shape[i] == 1) ? 0 : input_indices[i]; // if the target shape has a 1 then we want set the stride
            output_flat += output_idx * output_strides[i];                     // to 0 so that we can continuosly go to the beginning of 
        }                                                                      // that dimension
        
        result[output_flat] += input[input_flat];                   // now we have the correct input and output using the
                                                                    // indices of the input we can add to the result

        // start at the final dimension
        size_t dim = shape.size() - 1;
        while (dim < shape.size()) {
            input_indices[dim]++; // add the dimension of that input_indices automatically then if we need to set 0 later
            if (input_indices[dim] < shape[dim]) break; // if the input dim < shape dim, break
            input_indices[dim] = 0; // here the dim is greater than the max dim so loop back around to 0
            if (dim == 0) break;
            dim--; // decrease the dims again
        }
        
        // start with all_zero = true (input_indices)
        bool all_zero = true;
        for (size_t idx : input_indices) { // if one is nor 0 set all_zero = false
            if (idx != 0) { all_zero = false; break; }
        }
        if (all_zero && input_flat > 0) break; // leave the loop if all_zero (full gone through the loop)
        
    } while (true);
    
    return result;
}


Tensor TransposeOperation(Tensor& A) {

    std::vector<double>y(A.data->size(), 0.0);
    std::vector<size_t>newShape = A.shape;
    newShape[newShape.size() - 1] = A.shape[A.shape.size() - 2];
    newShape[newShape.size() - 2] = A.shape[A.shape.size() - 1]; // (2,4,3)
    
    std::vector<size_t> yStrides(newShape.size(), 1);
    std::partial_sum(newShape.rbegin(), newShape.rend() - 1, yStrides.rbegin() + 1, std::multiplies<>()); // (12,3,1)

    for (size_t n = 0; n < A.data->size(); n++) {
        size_t index = n;
        size_t element = 0;
        std::vector<size_t>normal_strides({});
        std::vector<size_t>op_strides({});
        
        for (auto iter = A.strides.begin(); iter != A.strides.end() - 2; ++iter) {
            int dimIdx = index / (*iter);
            index = index % (*iter);
            normal_strides.push_back(dimIdx);
        }

        if (A.strides.size() >= 2) {
            for (auto iter = A.strides.end() - 2; iter != A.strides.end(); ++iter) {
                int dimIdx = index / (*iter);
                index = index % (*iter);
                op_strides.push_back(dimIdx);
            }
        }

        std::vector<size_t>result_strides({});
        std::reverse(op_strides.begin(), op_strides.end());
        for (auto a : normal_strides) {
            result_strides.push_back(a);
        }
        for (auto a : op_strides) {
            result_strides.push_back(a);
        }

        for (int i = 0; i < yStrides.size(); i++) {
            element += (yStrides[i] * result_strides[i]);
        }
        y[element] = (*A.data)[n];
    };
    return Tensor(y, newShape, yStrides);
}

Tensor BasicMatMulOperation(Tensor& A, Tensor& B, std::function<double(double, double)> op) {
    MatMulIterator::MatMulIteratorConfig config(op);

    config.add_input(A);
    config.add_input(B);

    MatMulIterator iter = config.build();
    Tensor result = iter.for_each();
    return result;
}

void calc_gradient(Tensor& gradTensor,
std::vector<std::reference_wrapper<Tensor>>& save_tensors_,
std::function<double(double, double)> opA,
std::function<double(double, double)> opB) {

    Tensor& A = save_tensors_[0];
    Tensor& B = save_tensors_[1];

    auto Bchad = TransposeOperation(B);
    auto Achad = TransposeOperation(A);


    Tensor gradTensorA = BasicMatMulOperation(gradTensor, Bchad, [](double a, double b) -> double { return a * b; });
    Tensor gradTensorB = BasicMatMulOperation(Achad, gradTensor, [](double a, double b) -> double { return a * b; });

    // std::cout << "HERIE IS THE SHAPE: " << std::endl;
    // for (auto a : gradTensorA.shape) {
    //     std::cout << a << ", ";
    // }
    // for (auto a : gradTensorB.shape) {
    //     std::cout << a << ", ";
    // }

    std::vector<double> gradA = sum_to_shape(*gradTensorA.data, A.shape, gradTensorA.shape, gradTensorA.strides);
    std::vector<double> gradB = sum_to_shape(*gradTensorB.data, B.shape, gradTensorB.shape, gradTensorB.strides);

    for (size_t i = 0; i < A.data->size(); i++) {
        (*A.grad)[i] = (*A.grad)[i] + gradA[i];
    }

    // for (auto a : (*A.grad)) {
    //     std::cout << a << ", ";
    // }
    // for (auto a : (*B.grad)) {
    //     std::cout << a << ", ";
    // }
    
    for (size_t i = 0; i < B.grad->size(); i++) {
        (*B.grad)[i] = (*B.grad)[i] + gradB[i];
    }
}


void MatMulBackward::backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) { // grad_output is the gradient inputted into the function
    // std::cout << "MUL HAS BEEN CALLED" << std::endl;

    Tensor grad_tensor = Tensor(*grad_output, shape, stride);

    calc_gradient(
        grad_tensor,
        save_tensors_,
        [](double a, double b) -> double { return b; },
        [](double a, double b) -> double { return a; });
    
    for (Tensor& A : save_tensors_) {
        A.backward(false);
    }
}

void MatMulBackward::save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors) { //
    save_tensors_ = save_tensors;
};