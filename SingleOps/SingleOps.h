#pragma once

#include "../Tensor/Tensor.h"

Tensor ReLU(Tensor& A);

#include "../Iterator/SingleIterator/SingleIterator.h"
#include "../AutoGradEngine/AddBackwards.h"