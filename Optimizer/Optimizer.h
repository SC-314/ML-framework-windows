#include <vector>
#include <functional>
#include <initializer_list>

class Tensor;

class Optim {
public:
    virtual void apply_grads() = 0;
    virtual void zero_grads() = 0;
    virtual ~Optim() = default;
    std::vector<std::reference_wrapper<Tensor>> save_tensors_;
    
    class SGD;
};

class Optim::SGD : public Optim {
public:
    std::vector<Tensor*> param_ptrs_;  // Store pointers instead
    double learning_rate;
    
    SGD(double learning_rate, std::vector<Tensor*> param_ptrs);
    void apply_grads() override;
    void zero_grads() override;
};
#include "../Tensor/Tensor.h"