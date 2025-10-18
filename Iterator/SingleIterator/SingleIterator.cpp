#include "./SingleIterator.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cstdlib>

SingleIterator::SingleIterator(
    std::vector<Tensor> inputs,
    std::function<double(double)> op) : inputs(inputs), op(op) {}

SingleIterator::SingleIteratorConfig::SingleIteratorConfig(std::function<double(double)> op): op(op) {};

SingleIterator::SingleIteratorConfig& SingleIterator::SingleIteratorConfig::add_input(const Tensor& tensor) {
    inputs.push_back(tensor);
    return *this;
}

SingleIterator SingleIterator::SingleIteratorConfig::build() {

    return SingleIterator(inputs, op);
}

Tensor SingleIterator::for_each() {
    return loop(*this);
}