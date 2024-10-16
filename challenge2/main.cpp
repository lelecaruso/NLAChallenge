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

MatrixXd loadimg(const char* input_image_path);
void storeImg(MatrixXd img,int height, int width, const std::string output_image_path);

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;}

    const char* input_image_path = argv[1];
    MatrixXd img = loadimg(input_image_path);
    
    return 0;
}

MatrixXd loadimg(const char* input_image_path){
  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  
 cout << "Image loaded "<<endl;
  MatrixXd img(height,width);
    for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * 1;  // 1 channel as we are using greyscale
      img(i, j) = static_cast<double>(image_data[index])/(255); // we are from 0:255 to 0:1 
    }
  }
  stbi_image_free(image_data);
  return img;
}

void storeImg(MatrixXd img,int height, int width, const std::string output_image_path){
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    grayscale_image = img.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val*255);
    });

    // Save the grayscale image using stb_image_write
    if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                        grayscale_image.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;
    }
    else    std::cout << "\nImage saved to " << output_image_path << std::endl;

}