#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "../NeuralNetwork/Layer.h"

#pragma once
#include "../NeuralNetwork/Layer.h"
#include <vector>
#include <string>

class Module {
private:
    std::vector<std::pair<std::string, Layer*>> named_modules_;
    
public:
    void register_module(const std::string& name, Layer& layer) {
        named_modules_.emplace_back(name, &layer);
    }
    
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> all_params;
        for (auto& [name, layer_ptr] : named_modules_) {
            auto layer_params = layer_ptr->get_parameters();
            all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
        }
        return all_params;
    }
};