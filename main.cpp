#include <torch/script.h> // TorchScript头文件
#include <iostream>
#include <memory>

int main() {
    // Load model
    std::cout<<"step1: load model"<<std::endl;
    torch::jit::script::Module module =
      torch::jit::load("../model.pt");
    module.to(at::kCUDA);// if use CUDA
    std::cout<<module.dump_to_str(false,false,false)<<std::endl;
    // Generate a data
    std::cout<<"step2: generate random data input"<<std::endl;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, 100, 40}).to(at::kCUDA)); //if use CUDA
    //inputs.push_back(torch::rand({1, 1, 100, 40})); //if use CPU
    std::cout<<inputs<<std::endl;
    // Execute the model and turn its output into a tensor.
    auto output = module.forward(inputs).toTensor();
    std::tuple<torch::Tensor, torch::Tensor> predictions = torch::max(output, 1);
    auto prediction= std::get<1>(predictions);
    std::cout<<"step3: predict movement"<<std::endl;
    switch(prediction.item().toInt())
    {
        case 0 :
            std::cout << "up-movement" << std::endl;
        break;
        case 1 :
            std::cout << "stationary condition" << std::endl;
        break;
        case 2 :
           std::cout << "down-movement" << std::endl;
        break;
        default :
            std::cout << "invalid prediction" << std::endl;
    }
}