#include "./LossFunction.h"

MseLoss::MseLoss(Tensor& A, Tensor& B) { 
    Tensor im1 = (A - B);
    this->IM1 = im1;
    this->IM1.data = im1.data;
    this->IM1.grad_fn = im1.grad_fn;
    this->IM1.grad = im1.grad;

    Tensor im2 = (IM1 * IM1);
    this->IM2 = im2;
    this->IM2.data = im2.data;
    this->IM2.grad_fn = im2.grad_fn;
    this->IM2.grad = im2.grad;
}

void MseLoss::backward() {
    this->IM2.backward(true);
}