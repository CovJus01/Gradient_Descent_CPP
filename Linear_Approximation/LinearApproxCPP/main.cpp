#include "../../gradient-descent.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

int main() {

      // Learning Params and Data
      std::vector<double> x_train = {1.0, 2.0};
      std::vector<double> y_train = {300.0, 500.0};
      double init_w = 2.0;
      double init_b = 1.0;
      int iterations = 10000;
      double learning_rate = 0.01;

      // Usage
      auto cost = univariateCost<double>(init_w, init_b, x_train, y_train);
      std::cout << "Calculated Cost: "<< cost << std::endl;
      auto gradients = univariateGradient<double>(init_w, init_b, x_train, y_train);
      std::cout << "Calculated Gradients: "<< gradients.first << ", " << gradients.second << std::endl;
      auto final_params = univariateGradientDescent<double>(init_w, init_b, x_train, y_train, learning_rate, iterations);
      std::cout << "Calculated w & b: " << final_params.first << ", " << final_params.second << std::endl;
      return 0;
}