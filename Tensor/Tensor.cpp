#include "./Tensor.h"
#include <memory>


Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape, std::vector<size_t> strides):
    data(std::make_shared<std::vector<double>>(data)), shape(shape), strides(strides),
    grad(std::make_shared<std::vector<double>>(data.size(), 0.0)) {}

Tensor::Tensor():
    data(std::make_shared<std::vector<double>>(std::vector<double>({}))), shape(std::vector<size_t>({})),
    strides(std::vector<size_t>({})), grad(std::make_shared<std::vector<double>>()) {}

Tensor Tensor::operator=(const Tensor& other) {
    this->data = other.data;
    this->shape = other.shape;
    this->strides = other.strides;
    return *this;
}

Tensor Tensor::copy() {
    Tensor A = *this;
    A.data = std::make_shared<std::vector<double>>(*data);
    return A;
}

void Tensor::set_grad_fn(std::shared_ptr<GraphNode> node) {
    this->grad_fn = node;
}

Tensor Tensor::operator+(Tensor& other) {
    return add(*this, other);
}

Tensor Tensor::operator-(Tensor& other) {
    return sub(*this, other);
}

Tensor Tensor::operator*(Tensor& other) {
    return mul(*this, other);
}

Tensor Tensor::operator/(Tensor& other) {
    return div(*this, other);
}

Tensor Tensor::operator&(Tensor& other) {
    return matmul(*this, other);
}

void Tensor::backward(bool final) {
    if (final) {
        grad = std::make_shared<std::vector<double>>(std::vector<double>(data->size(), 1.0));
    }

    if (grad_fn != nullptr) {
        grad_fn->backward(grad, shape, strides);
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "[";
    for (auto& elem : (*tensor.data)) {
        os << elem << ", ";
    }
    return os << "]";
}