#include <Eigen/Dense>
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
MatrixXd createImgChess();
MatrixXd Noise(MatrixXd img, int height, int width);

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
    mpirun -n 4 ./eigen1 /home/jellyfish/shared-folder/NLAChallenge/challenge2/mat_market.mtx eigvec.txt hist.txt -etol 1.0e-8 -e pi

    number of processes = 4
    matrix size = 256 x 256 (65536 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR
    shift                 : 0.000000e+00
    eigensolver status    : normal end

    Power: mode number          = 0
    Power: eigenvalue           = 1.608332e+04
    Power: number of iterations = 8
    Power: elapsed time         = 1.168337e-03 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 1.866013e-09
   */

//Task 4
/*Find a shift μ ∈Ryielding an acceleration of the previous eigensolver. Report μ and the
number of iterations required to achieve a tolerance of 10−8.*/

//We have computed the smallest eigenvalue and then we find the shift s.t.  the max eigenvalue is the same as before, but we found it with less iterations. (7 vs 8)

  /*
  mpirun -n 4 ./eigen1 /home/jellyfish/shared-folder/NLAChallenge/challenge2/mat_market.mtx eigvec.txt hist.txt -etol 1.0e-8 -e pi -shift 1000

number of processes = 4
matrix size = 256 x 256 (65536 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
matrix storage format : CSR
shift                 : 1.000000e+03
eigensolver status    : normal end

Power: mode number          = 0
Power: eigenvalue           = 1.608332e+04
Power: number of iterations = 7
Power: elapsed time         = 4.309720e-04 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 6.868709e-09
  */

 //Waiting for SVD lab to be completed

 //Task 8
 /*Using Eigen create a black and white checkerboard image (as the one depicted below)
with height and width equal to 200 pixels. Report the Euclidean norm of the matrix
corresponding to the image*/

  MatrixXd chessboard = createImgChess();
  int dim_chess_h = chessboard.rows();
  int dim_chess_w = chessboard.cols();

  //Task 9 
  /*Introduce a noise into the checkerboard image by adding random fluctuations of color
  ranging between [−50,50] to each pixel. Export the resulting image in .png and upload it*/
  MatrixXd chessboard_noise = Noise(chessboard, dim_chess_h , dim_chess_w);

  //Missing tasks for SVD
    
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

MatrixXd createImgChess(){
  int dim = 25;
  MatrixXd black = MatrixXd:: Zero(dim,dim);
  MatrixXd white = MatrixXd:: Ones(dim,dim);

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
    MatrixXd noise = 50.0/255.0 * (MatrixXd::Random(height,width));
    MatrixXd img_noise = noise + img;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if(img_noise(i,j)>1) img_noise(i,j) = 1;
            if(img_noise(i,j)<0) img_noise(i,j) = 0;
        }
    }
storeImg(img_noise,height,width,"scacchiera_noise.png");
return img_noise;
}

