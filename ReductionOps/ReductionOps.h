#pragma once

#include "../Tensor/Tensor.h"

Tensor Sum(Tensor& A);

#include "../Iterator/ReductionIterator/ReductionIterator.h"
#include "../AutoGradEngine/AddBackwards.h"