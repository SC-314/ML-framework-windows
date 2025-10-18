#include "../ReductionOps/ReductionOps.h"
#include <memory>
#include <numeric>


class ReductionIterator;

Tensor Sum(Tensor& A) {
    ReductionIterator::ReductionIteratorConfig config(
        [](std::vector<double> a) -> double { return double(std::accumulate(a.begin(), a.end(), 0.0)); }
    );

    config.add_input(A);

    ReductionIterator iter = config.build();

    Tensor result = iter.for_each();

    // std::cout << "POSIEFJPOSIJ" << std::endl;

    auto node = std::make_shared<SumBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A}));
    result.grad_fn = node;

    return result;
}