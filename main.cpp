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
        Net() : fc1(2, 5), fc2(5,20), fc3(20, 1) {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("fc3", fc3);
        }
        Tensor forward(Tensor& x) {
            Tensor C = fc1(x);
            this->IM1 = C;
            this->IM1.data = C.data; // it didmt fix the data notbinmg in iter
            this->IM1.grad_fn = C.grad_fn;
            this->IM1.grad = C.grad;

            Tensor D = fc2(IM1);
            this->IM2 = D;
            this->IM2.data = D.data;
            this->IM2.grad_fn = D.grad_fn;
            this->IM2.grad = D.grad;

            return fc3(IM2);
        }
    };
    Net network;
    Optim::SGD optimizer(0.000001, network.parameters());
    Tensor A = Tensor(std::vector<double>({1,2,3,4,5,6,7,3,3,3}), std::vector<size_t>({5,2}), std::vector<size_t>({2,1}));
    Tensor y = Tensor(std::vector<double>({1,1,1,1,1}), std::vector<size_t>({5,1}), std::vector<size_t>({1,1}));


    for (int i = 0; i < 10000; i++) {
        Tensor B = network.forward(A);
        auto loss = MseLoss(B, y);
        loss.backward();
        optimizer.apply_grads();
        std::cout << B << std::endl;
        optimizer.zero_grads();
    }
}
