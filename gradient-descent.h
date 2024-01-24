#include <math.h>
#include <vector>

template <typename type>
type cost();

template <typename type>
type squared_error();

template <typename type>
type gradient_descent_single();

template <typename type>
type gradient_descent();

template <typename type>
type squared_error(type w, type b, std::pair<type, type> &expected) {
  return pow(((expected.first * w + b ) - expected.second), 2);
}

template <typename T>
T cost_linear(T w, T b,  std::vector<std::pair<T, T>> &expected) {
 T sum = 0;
 for (auto i : expected) {
  sum += squared_error<T>(w, b, expected);
 }

 return sum;
}


