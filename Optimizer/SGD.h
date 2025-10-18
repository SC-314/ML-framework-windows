#include "./Optimizer.h"

class SGD : public Optim {
public:
    SGD(std::vector<Tensor> param_list);
    void step() override {

    }
};