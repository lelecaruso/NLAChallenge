#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <iostream>
#include <cmath> // per sqrt()
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
MatrixXd compTruncation(MatrixXd A, int k, bool d);
MatrixXd filterColor(MatrixXd A);
MatrixXd createChessBoard();
MatrixXd noise(MatrixXd img, int height, int width);

int main(int argc, char* argv[]){
    if (argc < 2) 
        {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
        }
    const char* input_image_path = argv[1];
    MatrixXd A = loadimg(input_image_path);
    printf("Dimensions:\n%d x %d\n", A.rows(), A.cols());
    //1
    MatrixXd Cov = A.transpose()*A;
    printf("The norm of the covariance matrix is:\n%f \n", Cov.norm() );
    //2 since AT*A is symmetric we can use 
    SelfAdjointEigenSolver<MatrixXd> eigensolver(Cov);
    if (eigensolver.info() != Eigen::Success) abort();
    std::cout << "The two largest SV of A are:\n" << sqrt(eigensolver.eigenvalues().tail(2)[0]) << "\n" << sqrt(eigensolver.eigenvalues().tail(2)[1]) << std::endl;
    //3 
    Eigen::saveMarket(Cov, "mat_market.mtx");
    /*
      eigensolver           : Power
      convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
      Power: eigenvalue           = 1.045818e+09
      Power: number of iterations = 8
    */

    //4 shift = eigenvalue2 / 2
    /* 
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    shift                 : 4.568000e+07
    Power: eigenvalue           = 1.045818e+09
    Power: number of iterations = 7
    */

    //5
    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd E = svd.singularValues().asDiagonal();
    printf("Norm of the Sigma Matrix is:\n%f \n", E.norm() );
    //6
    MatrixXd C40 = compTruncation(svd.matrixU(), 40, true);
    MatrixXd C80 = compTruncation(svd.matrixU(), 80, true);
    MatrixXd D40 = compTruncation(svd.matrixV(), 40, true)*svd.singularValues().head(40).asDiagonal();
    MatrixXd D80 = compTruncation(svd.matrixV(), 80, true)*svd.singularValues().head(80).asDiagonal();
    printf("NonZeroes:\nC40 = %d , D40 = %d\nC80 = %d , D80 = %d\n",C40.nonZeros(), D40.nonZeros(), C80.nonZeros(), D80.nonZeros() );
    //7
    MatrixXd A40 = filterColor( C40*D40.transpose() );
    storeImg(A40, A40.rows(), A40.cols(), "./output_2/truncated40.png");
    MatrixXd A80 = filterColor( C80*D80.transpose() );
    storeImg(A80, A80.rows(), A80.cols(), "./output_2/truncated80.png");
    //8
    MatrixXd B = createChessBoard();
    printf("The Euclidian Norm of Chessboard:\n%f \n", B.norm());
    //9
    MatrixXd BN = noise(B, B.rows(), B.cols()) ;
    //10
    Eigen::BDCSVD<Eigen::MatrixXd> svd2 (BN, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd EN = svd2.singularValues().asDiagonal();
    cout<< "Two biggest singular Values are:\n"<<svd2.singularValues().head(2)<<endl;
    //11
    MatrixXd C5 = compTruncation(svd2.matrixU(), 5, true);
    MatrixXd C10 = compTruncation(svd2.matrixU(), 10, true);
    MatrixXd D5 = compTruncation(svd2.matrixV(), 5, true)*svd2.singularValues().head(5).asDiagonal();
    MatrixXd D10 = compTruncation(svd2.matrixV(), 10, true)*svd2.singularValues().head(10).asDiagonal();
    printf("Sizes:\nC5 = %d x %d , D5 = %d x %d \nC10 = %d x %d, D10 = %d x %d\n"
          ,C5.rows(), C5.cols(), D5.rows(), D5.cols(), C10.rows(), C10.cols(), D10.rows(), D10.cols() );
    //12
    MatrixXd A5 = filterColor( C5*D5.transpose() );
    storeImg(A5, A5.rows(), A5.cols(), "./output_2/chessT5.png");
    MatrixXd A10 = filterColor( C10*D10.transpose() );
    storeImg(A10, A10.rows(), A10.cols(), "./output_2/chessT10.png");  

    
    
        //Task 13
    /*Compare the compressed images with the original and noisy images. Comment the results.

The compressed images appear much closer to the original image than to the noisy one. 
This occurs because the first two singular values of the original image are on the order of 2.5*10^4 and 2.3*10^5, while the remaining  singular values are between the order of [10^2 and 10^0]. 
By compressing the image using only 5 or 10 singular values, the main details are preserved, while the high-frequency components representing noise are filtered out. 
The compressed image with 5 singular values is  even more similar to the original chessboard(no noise) than the one with 10, as it only includes the most important part of the image descarding the noise.
As a result, the compressed images visually resemble the original, as the compression highlights the dominant structural features without amplifying the noise components.

    
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
      img(i, j) = static_cast<double>(image_data[index]); 
    }
  }
  stbi_image_free(image_data);
  return img;
}

void storeImg(MatrixXd img,int height, int width, const std::string output_image_path){
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
    grayscale_image = img.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val);
    });

    // Save the grayscale image using stb_image_write
    if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                        grayscale_image.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;
    }
    else    std::cout << "\nImage saved to " << output_image_path << std::endl;

}

MatrixXd compTruncation(MatrixXd A, int k, bool dec){
  int rows = k,cols = k;
  if( dec ) {
    rows = A.rows();
    cols = k;
  }
  MatrixXd B(rows, cols);
  for( int i = 0; i < rows; i++ ){
    for( int j = 0; j < cols ; j++)
    {
      B(i,j) = A(i,j);
    }
  }
  return B;
}

MatrixXd filterColor(MatrixXd A){
  for(int i = 0; i < A.rows(); i++){
    for( int j = 0; j < A.cols(); j++){
      if( A(i,j) > 255 ) A(i,j) = 255;
      if( A(i,j) < 0 ) A(i,j) = 0;
    }
  }
  return A;
}

MatrixXd createChessBoard(){
    int dim = 25;
    MatrixXd black = MatrixXd:: Zero(dim,dim);
    MatrixXd white = MatrixXd:: Ones(dim,dim)*255;

    bool B_W = false; //false is black
    bool prec = false;

  int dim_chess = 200;
  MatrixXd matrix(dim_chess,dim_chess);
    for(int i = 0 ;i<8;i++){
      for(int j = 0; j<8; j++){

        if(B_W == false) {
          matrix.block(i*25, j*25, 25, 25) = black;
          prec = B_W;
          B_W = true;
        }
        else {
        matrix.block(i*25, j*25, 25, 25) = white;
        prec = B_W;
        B_W = false;
        }

      }
      B_W = prec;
    }
  storeImg(matrix,dim_chess,dim_chess,"./output_2/chessboard.png");
  return matrix;
}

MatrixXd noise(MatrixXd img, int height, int width){
  MatrixXd noise = 50.0 * (MatrixXd::Random(height,width));
  MatrixXd img_noise = filterColor(noise + img);
  storeImg(img_noise,height,width,"./output_2/noised.png");
  return img_noise;
}