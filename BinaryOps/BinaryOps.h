#pragma once

#include "../Tensor/Tensor.h"

Tensor add(Tensor& A, Tensor& B);
Tensor sub(Tensor& A, Tensor& B);
Tensor mul(Tensor& A, Tensor& B);
Tensor div(Tensor& A, Tensor& B);

#include "../Iterator/ElemWiseIterator/ElemWiseIterator.h"
#include "../Iterator/MatMulIterator/MatMulIterator.h"
#include "../AutoGradEngine/AddBackwards.h"