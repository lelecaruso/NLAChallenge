#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/Core>
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
MatrixXd createImgChess();
MatrixXd Noise(MatrixXd img, int height, int width);

MatrixXd reduction_k(int k, MatrixXd reduction_matrix);

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;}


    // Task 1
    /*
    Load the image as an Eigen matrix A with size m×n. Each entry in the matrix corresponds
    to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
    Compute the matrix product A^TA and report the euclidean norm of A^TA
    */

   const char* input_image_path = argv[1];
   MatrixXd img = loadimg(input_image_path);

   MatrixXd A_transpose = img.transpose();
   //Verifico le dimensioni della matrice 
   int rows = A_transpose.rows();
   int cols = A_transpose.cols();
   std::cout << "The transpose matrix has dimensions: " << rows << " x " << cols << std::endl;

   MatrixXd result = A_transpose * img; //salvo il risultato del prodotto 
   double matrix_norm = result.norm();
   std::cout << "The matrix has norm: " << matrix_norm << std::endl;

   //Task 2
   /*
   Solve the eigenvalue problem ATAx= λxusing the proper solver provided by the Eigen
   library. Report the two largest computed singular values of A.
   */

  // Since A^TA is a sym matrix we can use the SelfAdjointEigenSolver<> solver()
   SelfAdjointEigenSolver<MatrixXd> eigensolver(result);
   if (eigensolver.info() != Eigen::Success) abort();
    //std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
   VectorXd eigenvalues = eigensolver.eigenvalues(); //store the eigenvalues in a vector 
   std::vector<double> eigenvals(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
   std::cout << "The biggest eigenvalue of A^TA are:\n" << eigenvalues[eigenvals.size()-1] << std::endl;
   std::cout << "The second biggest eigenvalue of A^TA are:\n" << eigenvals[eigenvals.size() - 2 ] << std::endl;

   //since we have the eigenvalues of A^TA the singular values of A can be computed as: si = sqrt(lambda_i)
   std::cout << "The biggest singular value of A are:\n" << sqrt(eigenvalues[eigenvals.size()-1]) << std::endl;
   std::cout << "The second biggest singular value of A are:\n" << sqrt(eigenvals[eigenvals.size() - 2 ]) << std::endl;
  
  //Task 3
  /*
  Export matrix A^TA in the matrix market format and move it to the lis-2.1.6/test
  folder. Using the proper iterative solver available in the LIS library compute the largest
  eigenvalue of ATA up to a tolerance of 10−8. Report the computed eigenvalue. Is the result
  in agreement with the one obtained in the previous point?
  */

  SpMat mat_market(rows,rows);
  for(int i = 0; i < rows;i++){
    for(int j = 0; j < rows;j++){
      if(result(i,j)!=0){mat_market.coeffRef(i,j)=result(i,j);}
    }
  }
   Eigen::saveMarket(mat_market, "mat_market.mtx");

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


 //Task 5
 /*
 Using the SVD module of the Eigen library, perform a singular value decomposition of the
 matrix A. Report the Euclidean norm of the diagonal matrix Σ of the singular values.
 */
//img is the matrix A
 MatrixXd A = MatrixXd(img); // convert to dense for SVD
 Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
 MatrixXd sigma =  svd.singularValues().asDiagonal();
 double norm = sigma.norm();
 std::cout << "\nSigma norm is:  " << norm << std::endl;

 //Task 6
 /*
 (1) C = [u1 u2 . . . uk], D = [σ1v1 σ2v2 . . . σkvk]
 */
 /*Compute the matrices C and D described in (1) assuming k = 40 and k = 80. Report the
 number of nonzero entries in the matrices C and D.*/

 MatrixXd U = svd.matrixU();
 int rows1 = U.rows();
 int cols1 = U.cols();
 std::cout << "The matrix U associated to svd  has dimensions: " << rows1 << " x " << cols1 << std::endl;
 int k1 = 40;
 int k2 = 80;
 MatrixXd C_40(rows1,k1);
 int nonzeros_C40 = 0;

 for(int i = 0; i<rows1; i++){
  for(int j=0 ; j<k1;j++){
     if(U(i,j) != 0 ){
        nonzeros_C40 ++;
      }
      C_40(i,j) = U(i,j);
  }
 }
 int rows2 = C_40.rows();
 int cols2 = C_40.cols();
 std::cout << "The C_40 has dimensions: " << rows2 << " x " << cols2 << std::endl;

int nonzeros_C80 = 0;
MatrixXd C_80(rows1,k2);
 for(int i = 0; i<rows1; i++){
  for(int j=0 ; j<k2;j++){
      if(U(i,j) != 0 ){
        nonzeros_C80 ++;
      }
        C_80(i,j) = U(i,j);
  }
 }
 int rows3 = C_80.rows();
 int cols3 = C_80.cols();
 std::cout << "The C_80 has dimensions: " << rows3 << " x " << cols3 << std::endl;

 std::cout <<"Number of non zeros entries for C-40 are:"<<nonzeros_C40 << std:: endl;
 std::cout <<"Number of non zeros entries for C-80 are:"<<nonzeros_C80 << std:: endl;

 MatrixXd V = svd.matrixV();
 int rows_v = V.rows();
 int cols_v = V.cols();
 std::cout << "The V has dimensions: " << rows_v << " x " << cols_v << std::endl;
 MatrixXd D_40 = MatrixXd:: Zero(rows_v,k1);
 MatrixXd D_80 = MatrixXd:: Zero (rows_v,k2);

int nonzeros_D40 = 0;
int nonzeros_D80 = 0;

VectorXd sigma_40 = svd.singularValues().head(k1);  // Prende i primi k1 valori singolari
// Costruisci la matrice diagonale con asDiagonal()
MatrixXd sigma_40_diag = sigma_40.asDiagonal();

for(int i = 0; i<rows_v; i++){
  for(int j=0 ; j<k1;j++){
     if(V(i,j) != 0 ){
        nonzeros_D40 ++;
      }
      D_40(i,j) = V(i,j);
  }
 }

 D_40 = D_40 * sigma_40_diag;

 VectorXd sigma_80 = svd.singularValues().head(k2);  // Prende i primi k1 valori singolari
// Costruisci la matrice diagonale con asDiagonal()
MatrixXd sigma_80_diag = sigma_80.asDiagonal();

for(int i = 0; i<rows_v; i++){
  for(int j=0 ; j<k2;j++){
     if(V(i,j) != 0 ){
        nonzeros_D80 ++;
      }
      D_80(i,j) = V(i,j);
  }
 }

 D_80 = D_80 * sigma_80_diag;

 std::cout <<"Number of non zeros entries for D-40 are:"<<nonzeros_D40 << std:: endl;
 std::cout <<"Number of non zeros entries for D-80 are:"<<nonzeros_D80 << std:: endl;

 std::cout << "The D_40 has dimensions: " << D_40.rows() << " x " << D_40.cols() << std::endl;
 std::cout << "The D_80 has dimensions: " << D_80.rows() << " x " << D_80.cols() << std::endl;

 //Task 7
 /*
 Compute the compressed images as the matrix product CDT (again for k = 40 and k = 80).
 Export and upload the resulting images in .png
 */

MatrixXd result_40 = C_40 * D_40.transpose();

    for (int i = 0; i < result_40.rows(); ++i) {
        for (int j = 0; j < result_40.cols(); ++j) {
            if(result_40(i,j)>255) result_40(i,j) = 255;
            if(result_40(i,j)<0) result_40(i,j) = 0;
        }
    }
    
storeImg(result_40,result_40.rows(),result_40.cols(),"immagine_compressa_40.png");


MatrixXd result_80 = C_80 * D_80.transpose();

   for (int i = 0; i < result_80.rows(); ++i) {
        for (int j = 0; j < result_80.cols(); ++j) {
            if(result_80(i,j)>255) result_80(i,j) = 255;
            if(result_80(i,j)<0) result_80(i,j) = 0;
        }
    }


storeImg(result_80,result_80.rows(),result_80.cols(),"immagine_compressa_80.png");


 //Task 8
 /*Using Eigen create a black and white checkerboard image (as the one depicted below)
with height and width equal to 200 pixels. Report the Euclidean norm of the matrix
corresponding to the image*/

  MatrixXd chessboard = createImgChess();
  int dim_chess_h = chessboard.rows();
  int dim_chess_w = chessboard.cols();
  std::cout << "The chessboard-matrix has norm: " << chessboard.norm() << std::endl;

  //Task 9 
  /*Introduce a noise into the checkerboard image by adding random fluctuations of color
  ranging between [−50,50] to each pixel. Export the resulting image in .png and upload it*/
  MatrixXd chessboard_noise = Noise(chessboard, dim_chess_h , dim_chess_w);

  //Task 10
  /*
  Using the SVD module of the Eigen library, perform a singular value decomposition of the
  matrix corresponding to the noisy image. Report the two largest computed singular values.
  */
 Eigen::BDCSVD<Eigen::MatrixXd> svd_chess (chessboard_noise, Eigen::ComputeThinU | Eigen::ComputeThinV);
 VectorXd W = svd_chess.singularValues();
 std::cout << "biggest singular values: " << W(0) << "\n";
 std::cout << "2nd biggest singular values: " << W(1) << "\n";

 std::cout << "all singular values are: " << W << "\n";

 //Task 11
 /*
 Starting from the previously computed SVD, create the matrices C and D defined in (1)
assuming k = 5 and k = 10. Report the size of the matrices C and D
 */
MatrixXd C_5 = reduction_k(5,svd_chess.matrixU());
MatrixXd C_10 = reduction_k(10,svd_chess.matrixU());

 MatrixXd V_chess = svd_chess.matrixV();
 int rows_v_chess = V_chess.rows();
 int cols_v_chess = V_chess.cols();
 int k1_chess = 5;
 int k2_chess = 10;
 std::cout << "The V has dimensions: " << rows_v_chess << " x " << cols_v_chess << std::endl;
 MatrixXd D_5 = MatrixXd:: Zero(rows_v_chess,k1_chess);
 MatrixXd D_10 = MatrixXd:: Zero (rows_v_chess,k2_chess);

int nonzeros_D5 = 0;
int nonzeros_D10 = 0;

VectorXd sigma_5 = svd_chess.singularValues().head(k1_chess);  // Prende i primi k1 valori singolari
// Costruisci la matrice diagonale con asDiagonal()
MatrixXd sigma_5_diag = sigma_5.asDiagonal();


for(int i = 0; i<rows_v_chess; i++){
  for(int j=0 ; j<k1_chess;j++){
     if(V(i,j) != 0 ){
        nonzeros_D5 ++;
      }
      D_5(i,j) = V_chess(i,j);
  }
 }

 D_5 = D_5 * sigma_5_diag;

 VectorXd sigma_10 = svd_chess.singularValues().head(k2_chess);  // Prende i primi k1 valori singolari
// Costruisci la matrice diagonale con asDiagonal()
MatrixXd sigma_10_diag = sigma_10.asDiagonal();

for(int i = 0; i<rows_v_chess; i++){
  for(int j=0 ; j<k2_chess;j++){
     if(V(i,j) != 0 ){
        nonzeros_D10 ++;
      }
      D_10(i,j) = V_chess(i,j);
  }
 }

 D_10 = D_10 * sigma_10_diag;
 std::cout << "The C5 has dimensions: " << C_5.rows() << " x " << C_5.cols() << std::endl; 
 std::cout << "The C10 has dimensions: " << C_10.rows() << " x " << C_10.cols() << std::endl;
 std::cout << "The D5 has dimensions: " << D_5.rows()<< " x " << D_10.cols() << std::endl;
 std::cout << "The D10 has dimensions: " << D_10.rows() << " x " << D_10.cols() << std::endl;

//Task 12
/*
Compute the compressed images as the matrix product CDT (again for k = 5 and k = 10).
Export and upload the resulting images in .png.
*/

MatrixXd result_5 = C_5 * D_5.transpose();

    for (int i = 0; i < result_5.rows(); ++i) {
        for (int j = 0; j < result_5.cols(); ++j) {
            if(result_5(i,j)>255) result_5(i,j) = 255;
            if(result_5(i,j)<0) result_5(i,j) = 0;
        }
    }
    
storeImg(result_5,result_5.rows(),result_5.cols(),"scacchiera_noise_compressa_5.png");


MatrixXd result_10 = C_10* D_10.transpose();

   for (int i = 0; i < result_10.rows(); ++i) {
        for (int j = 0; j < result_10.cols(); ++j) {
            if(result_10(i,j)>255) result_10(i,j) = 255;
            if(result_10(i,j)<0) result_10(i,j) = 0;
        }
    }
    
storeImg(result_10,result_10.rows(),result_10.cols(),"scacchiera_noise_compressa_10.png");

    
    //Task 13
    /*Compare the compressed images with the original and noisy images. Comment the results.

The compressed images appear much closer to the original image than to the noisy one. 
This occurs because the first two singular values of the original image are on the order of 2.5*10^4 and 2.3*10^5, while the remaining  singular values are between the order of [10^2 and 10^0]. 
By compressing the image using only 5 or 10 singular values, the main details are preserved, while the high-frequency components representing noise are filtered out. 
The compressed image with 5 singular values is  even more similar to the original chessboard(no noise) than the one with 10, as it only includes the most important part of the image descarding the noise.
As a result, the compressed images visually resemble the original, as the compression highlights the dominant structural features without amplifying the noise components.
*/
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
      img(i, j) = static_cast<double>(image_data[index]); // we are 0:255 
    }
  }
  stbi_image_free(image_data);
  return img;
}

void storeImg(MatrixXd img,int height, int width, const std::string output_image_path){
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
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

MatrixXd createImgChess(){
  int dim = 25;
  MatrixXd black = MatrixXd:: Zero(dim,dim);
  MatrixXd white = MatrixXd:: Ones(dim,dim);
  white = white * 255;
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
storeImg(matrix,dim_chess,dim_chess,"scacchiera.png");
return matrix;
}

MatrixXd Noise(MatrixXd img, int height, int width){
    MatrixXd noise = 50.0 * (MatrixXd::Random(height,width));
    MatrixXd img_noise = noise + img;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if(img_noise(i,j)>255) img_noise(i,j) = 255;
            if(img_noise(i,j)<0) img_noise(i,j) = 0;
        }
    }
    
storeImg(img_noise,height,width,"scacchiera_noise.png");
return img_noise;
}


MatrixXd reduction_k(int k, MatrixXd reduction_matrix){
  MatrixXd tmp = reduction_matrix.leftCols(k);
  return tmp;
}

