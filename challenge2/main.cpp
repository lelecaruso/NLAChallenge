#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
typedef Eigen::Triplet<double> T;
using SpMat=Eigen::SparseMatrix<double,Eigen::RowMajor>;
using SpVec=Eigen::VectorXd;
using namespace Eigen; //shorcut for eigen declaration


int main(int argc, char* argv[]) {

  //Task 1. Load the image as an Eigen matrix with size m × n. Each entry in the matrix corresponds
  //to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
  //Report the size of the matrix
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  const char* input_image_path = argv[1];

  // Load the image using stb_image
  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  // Since we have a greyscale image we use only 1 channel
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path << std::endl;
    return 1;
  }

 cout << "Image loaded "<<endl;

  MatrixXd img(height,width);
  
  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * 1;  // 1 channel as we are using greyscale
      img(i, j) = static_cast<double>(image_data[index])/(255); // we are scaling from 0:255 to 0:1 
    }
  }

  // Free memory
  stbi_image_free(image_data);

  return 0;
}