# Making a neural network in C++ (from scratch)

## What can it do?
The neural can do basic regression between input and solution data. The neural network includes linear layers, ReLU, MSE. You can also combine any Tensors you want due to the dynamic compute graph, this will be explored in later sections.

### Overview of main part
```c++
  Net network; // define neural network
  Optim::SGD optimizer(0.001, network.parameters()); // Define optimizer, input learning rate
  Tensor X = Tensor(std::vector<double>({-1,2,-3,4,-2,1,1,-3,3,3}), // X (input data)
  std::vector<size_t>({5,2}), std::vector<size_t>({2,1}));
  Tensor y = Tensor(std::vector<double>({1,2,3,4,5}), // y (solution data)
  std::vector<size_t>({5,1}), std::vector<size_t>({1,1}));

  for (int i = 0; i < 100; i++) {
      Tensor B = network.forward(X); // forward part
      auto loss = MseLoss(B, y); // calculate the Loss
      loss.backward(); // backpropagate throught the network
      optimizer.apply_grads(); // add the gradients to the current weights
      optimizer.zero_grads(); // set gradients assigned to the weights back to 0
      if (i % 10 == 0) {
          std::cout << i << ") Loss value:\t  " << loss.value() << "\tCurrent output: " << B << std::endl;
      }
  }

```

## Example
In this example I just to show the neural network works I input data: X = {{-1,2},{-3,4},{-2,1},{1,-3},{3,3}} and fit it to y = {{1},{2},{3},{4},{5}}. This X is 5 data points with each point having 2 features. We are going to train a neural network defined by: Linear(2,7),ReLU, Linear(7,20), ReLU, Linear(20,1). Where Linear(a,b) is a linear layer with a matrix of weights of size (a,b) with a bias being added of size (1,b). The picture below shows this neural network visually.

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/1df9f5f8-3588-4f58-897e-c98d85287f9d" />

As you can see the output of our neural network appraoches {1,2,3,4,5}, and our Loss also falls to 0. 

<img width="1839" height="524" alt="image" src="https://github.com/user-attachments/assets/ecf34808-0613-489f-8725-1795ef7929c4" /> 


## Technical details
The setup of this code, will be split into 3 main sections: Tensor, Broadcasting, Compute Graph.

### Tensor
#### Contiguous vs multidimensional arrays (for storing Tensor data)
If you had a matrix [[1,2],[3,4]], then it might look simpler to keep it in this form, but in practice this is very slow and will make basic operations like adding two matrices or braodcasting very hard. Instead our tensor class will store a contiguous array [1,2,3,4], and then the Tensor data type will also store the shape and strides in a different vector. Now if we wanted to add to tensors together this will very easy to do and just a 1-to-1 addition for each index.

My Tensor class contains variables like; requires_grad if its requires gradient to be tracked, the grad_fn is a shared pointer for the operation-class that created the tensor, and ofcourse the data, shape, strides. Finally operators are also defined so we can use normal operations like + instead of A.add(B) for example.
#### Strides
Strides define how you would travel through the dimensions , for example: [[1,2,3],[4,5,6]] would have strides (3,1), because in the contiguous array [1,2,3,4,5,6] each single index increase moves through the 1st dimension once, and each time the cont... array index increases by 3, our index of the 2nd dimension in the matrix increases by 1. This will be essential for broadcasting.
#### My Tensor
```c++
class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::shared_ptr<std::vector<double>> data;
    std::shared_ptr<std::vector<double>> grad;
    std::shared_ptr<GraphNode> grad_fn;
    bool requires_grad = true;

    Tensor(std::vector<double> data, std::vector<size_t> shape, std::vector<size_t> strides);

    Tensor operator=(const Tensor& other);
    Tensor copy();
    Tensor operator+(Tensor& other);
    Tensor operator-(Tensor& other);
};
```
### Broadcasting
Broadcasting is an essential part of a neural network because it allows you to apply binary operaions to differently sized vectors.  For example; [[1,2],[3,4]] + [[1],[2]] here we have a (2,2) and a (2,1) being added which would throw an error except we have broadcasting so the summation turns to: [[1,2],[3,4]]+[[1,1],[2,2]] where the 1 and 2 have being duplicated across the row. This feature is directly used in the linear layer because the bias weight has to be copied and added to each row of the output (a,b) + (1,b), the (1,b) bias will be duplicated across the columns and then added. In this example below you can see each vector being copied across the rows and columns.

<img width="512" height="168" alt="image" src="https://github.com/user-attachments/assets/c0e031dd-ad5f-4944-a717-a79620b4a7bc" />

In practice you dont actually copy the vector over rows eg, because this would make the already huge million dimensional vectors even bigger that could crash the program. So instead we use operation strides that control the index of our contiguous array by repeating the index through different dimensions. If you are adding:
[[1,2]] and [[1,2],[2,3]], then we will create two different indices and when we add the two contiguous arrays: [1,2] and [1,2,3,4], we will travel through [1,2] twice. This is why we use strides because we can set the strides of the [1,2] to (0,1) instead of (2,1), and for [[1,2],[3,4]] keep as (2,1). This way for the [1,2] we dont increase the contiguous index by 2 because of the zero (0,1), but the [[1,2],[3,4]] with strides (2,1) will increase contiguous array index by 2.

#### My Broadcasting function for elementwise binary operations
```c++
Tensor loop(ElemWiseIterator iter) {
    int totalElements = 1;
    for (size_t i = 0; i < iter.resultShape.size(); i++) { // Calculate the total number of values
        totalElements *= iter.resultShape[i];              // in the tensor data
    }

    std::vector<double>result({});

    for (int n = 0; n < totalElements; n++) {
        int index = n;

        size_t elemA = 0;
        size_t elemB = 0;

        for (size_t idx = 0; idx < iter.resultShape.size(); idx++){ // Loop through each dimension of our resulting array
            int dimIdx = index / iter.operationStrides[2][idx]; // These two lines of code calculate the index through each
            index = index % iter.operationStrides[2][idx];      // dimension of the resulting shape

            elemA += (dimIdx * iter.operationStrides[0][idx]); // Using the index of the dimension we multiply by the
            elemB += (dimIdx * iter.operationStrides[1][idx]); // operation strides that gives the correct index
        }                                                      // of that dimension
        // Once we get the correct indices for A and B, we apply the operation
        // saved in the anonymouse function.
        double value = iter.op((*iter.inputs[0].data)[elemA],(*iter.inputs[1].data)[elemB]);
        result.push_back(value);
    }

    return Tensor(result, iter.resultShape, iter.operationStrides[2]);
}
```
### Compute Graph
The compute graph runs underneath the neural network and tracks any operations you do like if you matrix multiply C = A X B, then the compute graph will store a matrix multiply class object. In this object it will store the matrices address in memory. The multiply class is a node and its address is stored in the output C, so that when back-propagating from C we can apply the correct transformations to the current gradient and continue the chain to A and B. So that the compute graph could function, I stored the data of the tensors in the heap, the heap is just memory that wont be automatically deleted when you leave the scope of a function. We need this so our compute graph can access all the used tensors in the forward process, to then reverse through them all again. In particular I use smart-pointers, these are a wrapper of pointers that includes a reference counter so once nothing is referencing the object in memory it gets deleted.

### Example
The code for the graphnode for adding
```c++
class AddBackward : public GraphNode { // this is the backward virtual class part of all the graph nodes.
public:
    std::vector<std::reference_wrapper<Tensor>> save_tensors_;
    void backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) override;
    void save_for_backward(std::vector<std::reference_wrapper<Tensor>> save_tensors_) override;
};

// This is the backpropagation function for the element-wise division
void DivBackward::backward(std::shared_ptr<std::vector<double>> grad_output, std::vector<size_t> shape, std::vector<size_t> stride) {
    // std::cout << "DIV HAS BEEN CALLED" << std::endl;

    Tensor grad_tensor = Tensor(*grad_output, shape, stride);

    calc_gradient_basic(
        grad_tensor,
        save_tensors_,
        [](double a, double b) -> double { return 1/b; },         // Here we have formulas for gradients for a and b that are inputted as anonymous functions
        [](double a, double b) -> double { return -a/(b * b); }); // and this function will be applied to the grad_tensor, that was backpropagated
    
    for (Tensor& A : save_tensors_) {
        A.backward(false);
    }
}
```

## What I learnt?
Coding this in C++, absolutely helped me learn how to use pointers, classes, virtual classes, header files, anonymous functions. The hardest part of the coding was easily the shear amount of files that all had different purposes where if one part didnt work then the compute graph would not be able to record all the operations.
Other than C++, my mathematics in the area of coding also improved, calculating the gradients of the different operations I included, index math of broadcasting. The hardest part of the maths was the indexing and understanding how broadcasting could add two differently sized matrices. Coding the backpropagating through a broadcasting took a lot of time to workout, due to having to sum over specific dimensions to reverse the process of broadcasting. Backpropagating through the matrix multiply was quite hard because I had to transpose each of the matrices.
