#include "./ReductionIterator.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cstdlib>

ReductionIterator::ReductionIterator(
    std::vector<Tensor> inputs,
    std::function<double(std::vector<double>)> op) : inputs(inputs), op(op) {}

ReductionIterator::ReductionIteratorConfig::ReductionIteratorConfig(std::function<double(std::vector<double>)> op): op(op) {};

ReductionIterator::ReductionIteratorConfig& ReductionIterator::ReductionIteratorConfig::add_input(const Tensor& tensor) {
    inputs.push_back(tensor);
    return *this;
}

ReductionIterator ReductionIterator::ReductionIteratorConfig::build() {

    return ReductionIterator(inputs, op);
}

Tensor ReductionIterator::for_each() {
    return loop(*this);
}