#include "gradient-descent.h"
#include <iostream>
#include <vector>

int main() {
      std::vector<float> x_train = {1.0f, 2.0f};
      std::vector<float> y_train = {300.0f, 500.0f};
      float init_w = 2.0f;
      float init_b = 1.0f;
      float cost = univariateCost<float>(init_w, init_b, x_train, y_train);
      std::cout << "Calculated Cost: "<< cost << std::endl;
      auto gradients = univariateGradient<float>(init_w, init_b, x_train, y_train);
      std::cout << "Calculated Gradients: "<< gradients.first << ", " << gradients.second << std::endl;
      return 0;
}