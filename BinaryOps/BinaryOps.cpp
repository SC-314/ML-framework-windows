#include "../BinaryOps/BinaryOps.h"
#include <memory>

class ElemWiseIterator;

Tensor add(Tensor& A, Tensor& B) {
    ElemWiseIterator::ElemWiseIteratorConfig config(
        [](double a, double b) -> double { return a + b; }
    );

    config.add_input(A);
    config.add_input(B);

    ElemWiseIterator iter = config.build();

    Tensor result = iter.for_each();

    auto node = std::make_shared<AddBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A , B}));
    result.grad_fn = node;

    return result;
}

Tensor sub(Tensor& A, Tensor& B) {
    ElemWiseIterator::ElemWiseIteratorConfig config(
        [](double a, double b) -> double { return a - b; }
    );

    config.add_input(A);
    config.add_input(B);

    ElemWiseIterator iter = config.build();

    Tensor result = iter.for_each();

    auto node = std::make_shared<SubBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A , B}));
    result.grad_fn = node;

    return result;
}

Tensor mul(Tensor& A, Tensor& B) {
    ElemWiseIterator::ElemWiseIteratorConfig config(
        [](double a, double b) -> double { return a * b; }
    );

    config.add_input(A);
    config.add_input(B);

    ElemWiseIterator iter = config.build();

    Tensor result = iter.for_each();

    auto node = std::make_shared<MulBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A , B}));
    result.grad_fn = node;

    return result;
}

Tensor div(Tensor& A, Tensor& B) {
    ElemWiseIterator::ElemWiseIteratorConfig config(
        [](double a, double b) -> double { return a / b; }
    );

    config.add_input(A);
    config.add_input(B);

    ElemWiseIterator iter = config.build();

    Tensor result = iter.for_each();

    auto node = std::make_shared<DivBackward>();
    node->save_for_backward(std::vector<std::reference_wrapper<Tensor>>({A , B}));
    result.grad_fn = node;

    return result;
}