#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
typedef Eigen::Triplet<double> T;
using SpMat=Eigen::SparseMatrix<double,Eigen::RowMajor>;
using namespace Eigen; //shorcut for eigen declaration

VectorXd filter_grey(VectorXd v);
MatrixXd Noise(MatrixXd img, int height, int width);
VectorXd Mat2Vec(MatrixXd mat, int height, int width);
SparseMatrix<double,RowMajor> SmoothingMatrix(int h, int w);
MatrixXd Vec2Mat(VectorXd vec, int height, int width, const std::string output_image_path);
SparseMatrix<double,RowMajor> SharpeningMatrix(int h, int w);

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

 cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." <<  endl;

  MatrixXd img(height,width);
  
  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * 1;  // 1 channel as we are using greyscale
      img(i, j) = static_cast<double>(image_data[index]); 
    }
  }

  // Free memory
  stbi_image_free(image_data);

  //Task 1 We are reporting the size of the matrix
  cout <<"the number of rows is: "<<height<< endl;
  cout <<"the number of cols is: "<<width<< endl;
  //Task 2
  MatrixXd img_noised = Noise(img,height,width);
  //Task 3
  VectorXd v = Mat2Vec(img,height,width);
  VectorXd w = Mat2Vec(img_noised,height,width);
  cout<< "Norm of the vector v is: "<< v.norm()<<endl;
  //Task 4
  SparseMatrix<double,RowMajor> A1 = SmoothingMatrix(height,width);
  VectorXd multipl = filter_grey(A1*v);
  //Task 5
  VectorXd filtered_img = filter_grey(A1*w);
  Vec2Mat(filtered_img, height,width,"output_smoothed.png");
  //Task 6
  SparseMatrix<double,RowMajor> A2 = SharpeningMatrix(height,width);
  VectorXd sharpened_img = filter_grey(A2*v);
  SparseMatrix<double,RowMajor> A2_T = A2.transpose();
  SparseMatrix<double,RowMajor>  A2_diff = A2_T - A2;
  if( A2_diff.norm() ) cout<<"\nMatrix is NOT Symmetric";
  else cout<<"\nMatrix IS Symmetric";
  //Task 7
  Vec2Mat(sharpened_img, height,width,"output_sharpened.png");

  return 0;
}

VectorXd filter_grey(VectorXd v){
  for(int i = 0; i < v.size() ; i++){
    if( v(i) > 255 ) v(i) = 255;
    if( v(i) < 0 ) v(i) =0;
  }
  return v;
}


//Task N.2 Introduce a noise signal into the loaded image by adding random fluctuations of color
// ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it
MatrixXd Noise(MatrixXd img, int height, int width){
    MatrixXd noise = 50 * (MatrixXd::Random(height,width));
    MatrixXd img_noise = noise + img;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if(img_noise(i,j)>255) img_noise(i,j) = 255;
            if(img_noise(i,j)<0) img_noise(i,j) = 0;
        }
    }



  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
  // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
  grayscale_image = img_noise.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });

  // Save the grayscale image using stb_image_write
  const std::string output_image_path = "output_noised.png";
  if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                     grayscale_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
  }
    std::cout << "Noised image saved to " << output_image_path << std::endl;

    return img_noise;
}


// Task N.3 Reshape the original and noisy images as vectors v and w, respectively. Verify that each
// vector has m n components. Report here the Euclidean norm of v.
VectorXd Mat2Vec(MatrixXd mat, int height, int width){
    VectorXd vec(width*height);
    int index;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
        index = (i * width + j); 
        vec[index] = mat(i,j);
        }
    }
  return vec;
}


//Task N.4 Write the convolution operation corresponding to the smoothing kernel Hav2 as a matrix
//vector multiplication between a matrix A1 having size mn × mn and the image vector.
//Report the number of non-zero entries in A1.
SparseMatrix<double,RowMajor> SmoothingMatrix(int h, int w){

  SparseMatrix<double,RowMajor> mat(h*w,w*h); 
    int index = 0;
    double value = 1.0/9.0;
    for(int x = 0; x < h ; x++){
      for(int y = 0; y < w; y++){
        index = x*w + y;
        mat.coeffRef(index ,  index) = value; 
        if(index > 0) mat.coeffRef(index, index-1) = value;
        if(index < h*w-1) mat.coeffRef(index, index+1) = value;	
        if((x-1)*w + y > 0) mat.coeffRef(index, (x-1)*w+y) = value;
        if((x+1)*w + y < h*w-1) mat.coeffRef(index, (x+1)*w+y) = value;	
        if((x+1)*w + y + 1 < h*w - 1 ) mat.coeffRef(index, (x+1)*w+y+1) = value;	
        if((x-1)*w + y - 1 > 0 ) mat.coeffRef(index, (x-1)*w+y-1) = value;	
        if((x+1)*w + y - 1 < h*w - 1 && (x+1)*w + y - 1 > 0  ) mat.coeffRef(index, (x+1)*w+y-1) = value;	
        if((x-1)*w + y + 1 > 0 && (x+1)*w + y - 1 < h*w-1  ) mat.coeffRef(index, (x-1)*w+y+1) = value;	
      }
    }

  cout<< "Number of Non Zeros : "<< mat.nonZeros();

  return mat;
}


//Task N.5/7 Apply the previous smoothing filter to the noisy image by performing the matrix vector
//multiplication A1w. Export and upload the resulting image.
MatrixXd Vec2Mat(VectorXd vec, int height, int width,const std::string output_image_path){

    MatrixXd mat(height,width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
        mat(i,j) = vec[i * width + j];
        }
    }

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
  grayscale_image = mat.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });

  // Save the grayscale image using stb_image_write


  if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                     grayscale_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
  }
    std::cout << "\nFiltered image saved to " << output_image_path << std::endl;

    
  return mat;
}


//Task N.6 Write the convolution operation corresponding to the sharpening kernel Hsh2 as a matrix
//vector multiplication by a matrix A2 having size mn × mn. Report the number of non-zero
//entries in A2. Is A2 symmetric?

//DA CONTROLLARE GLI INDICI E GLI IF
SparseMatrix<double,RowMajor> SharpeningMatrix(int h, int w){
  SparseMatrix<double,RowMajor> mat(h*w,w*h); 
    int index = 0;
    for(int x = 0; x < h ; x++){
      for(int y = 0; y < w; y++){
        index = x*w + y;
        mat.coeffRef(index ,  index) = 9.0; 
        if(index > 0) mat.coeffRef(index, index-1) = -1.0;
        if(index+1 < h*w) mat.coeffRef(index, index+1) = -3.0;	
        if((x-1)*w + y > 0) mat.coeffRef(index, (x-1)*w+y) = -3.0;
        if((x+1)*w + y < h*w) mat.coeffRef(index, (x+1)*w+y) = -1.0;	
      }
    }

  cout<< "Number of Non Zeros : "<< mat.nonZeros();

  return mat;
}