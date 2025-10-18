#include "../Iterator/MatMulIterator/MatMulIterator.h"

class MatMulIterator;

Tensor matmul(Tensor& A, Tensor& B) {
    MatMulIterator::MatMulIteratorConfig config(
        [](double a, double b) -> double { return a + b; }
    );

    config.add_input(A);
    config.add_input(B);

    MatMulIterator iter = config.build();

    Tensor result = iter.for_each();

    auto node = std::make_shared<MatMulBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A , B}));
    result.set_grad_fn(node);
    
    return result;
}
