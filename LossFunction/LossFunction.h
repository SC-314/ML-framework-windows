#pragma once
#include "../Tensor/Tensor.h"
#include <cstddef>

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual void backward() = 0;
};

class MseLoss : public LossFunction {
public:
    Tensor X;
    Tensor y;
    Tensor Loss;

    Tensor IM1;
    Tensor IM2;

    MseLoss(Tensor& A, Tensor& B);

    void backward();
};