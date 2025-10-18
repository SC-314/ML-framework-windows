#include "../SingleOps/SingleOps.h"
#include <memory>

class SingleIterator;

Tensor ReLU(Tensor& A) {
    SingleIterator::SingleIteratorConfig config(
        [](double a) -> double { if (a > 0) {return a;} else {return 0;}; }
    );


    config.add_input(A);

    SingleIterator iter = config.build();

    

    Tensor result = iter.for_each();

    auto node = std::make_shared<ReLUBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A}));
    result.grad_fn = node;

    return result;
}
