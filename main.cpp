#include "Main/Main.h"
#include "LossFunction/LossFunction.h"
#include <sys/types.h> // infuture have tensor just be a class just to hold the singel poinpter to the
#include <vector>// tensorimpl that store all the data (store nothing on stack


int main() {

    struct Net : Module {
        Linear fc1{};
        Linear fc2{};
        Linear fc3{};
        Tensor IM1;
        Tensor IM2;
        Tensor IM3;
        Tensor IM4;
        Net() : fc1(2, 5), fc2(5,20), fc3(20, 1) {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("fc3", fc3);
        }
        Tensor forward(Tensor& x) {
            Tensor A = fc1(x);
            this->IM1 = A;
            this->IM1.data = A.data; // it didnt fix the data not being in iter
            this->IM1.grad_fn = A.grad_fn;
            this->IM1.grad = A.grad;

            Tensor Arelu = IM1.relu();
            this->IM2 = Arelu;
            this->IM2.data = Arelu.data;
            this->IM2.grad_fn = Arelu.grad_fn;
            this->IM2.grad = Arelu.grad;

            Tensor B = fc2(IM2);
            this->IM3 = B;
            this->IM3.data = B.data;
            this->IM3.grad_fn = B.grad_fn;
            this->IM3.grad = B.grad;

            Tensor Brelu = IM3.relu();
            this->IM4 = Brelu;
            this->IM4.data = Brelu.data;
            this->IM4.grad_fn = Brelu.grad_fn;
            this->IM4.grad = Brelu.grad;

            return fc3(IM4);
        }
    };
    Net network;
    Optim::SGD optimizer(0.001, network.parameters());
    Tensor A = Tensor(std::vector<double>({-1,2,-3,4,-2,1,1,-3,3,3}), std::vector<size_t>({5,2}), std::vector<size_t>({2,1}));
    Tensor y = Tensor(std::vector<double>({1,2,3,4,5}), std::vector<size_t>({5,1}), std::vector<size_t>({1,1}));

    for (int i = 0; i < 100; i++) {
        Tensor B = network.forward(A);
        auto loss = MseLoss(B, y);
        loss.backward();
        std::cout << B << std::endl;
        optimizer.apply_grads();
        optimizer.zero_grads();
    }

}
