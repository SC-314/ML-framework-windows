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

    Tensor im3 = IM2.sum();
    this->IM3 = im3;
    this->IM3.data = im3.data;
    this->IM3.grad_fn = im3.grad_fn;
    this->IM3.grad = im3.grad;

    Tensor n = Tensor(std::vector<double>({(*A.data).size()}), std::vector<size_t>({1}), std::vector<size_t>({1}));
    this->N = n;
    this->N.data = n.data;
    this->N.grad_fn = n.grad_fn;
    this->N.grad = n.grad;

}

void MseLoss::backward() {
    this->IM2.backward(true);
}