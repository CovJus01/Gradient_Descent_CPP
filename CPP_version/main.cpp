#include "gradient-descent.h"
#include <iostream>
#include <vector>

int main() {
      std::pair<float, float> d1 = {1.0f, 300.0f};
      std::pair<float, float> d2 = {2.0f, 500.0f};
      std::vector<std::pair<float, float>> dataset = {d1,d2};
      float cost = compute_univariate_cost<float>(2.0f, 1.0f, dataset);
      std::cout << "Calculated Cost: "<< cost << std::endl;
      return 0;
}