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
    Tensor IM3;
    Tensor IM4;

    Tensor N;

    MseLoss(Tensor& A, Tensor& B);

    void backward();

    double value() {
        double a =  (*IM1.data).size();
        return (*IM3.data)[0] / a;
    }
};