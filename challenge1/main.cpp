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
using SpVec=Eigen::VectorXd;
using namespace Eigen; //shorcut for eigen declaration

VectorXd filter_grey(VectorXd v);
void isSparseSymmetric(SparseMatrix<double,RowMajor> A2, const std::string mn);
SparseMatrix<double,RowMajor> createConvolution(std::vector<double> values, int height, int width );
MatrixXd Noise(MatrixXd img, int height, int width);
VectorXd Mat2Vec(MatrixXd mat, int height, int width);
SparseMatrix<double,RowMajor> SmoothingMatrix(int h, int w);
MatrixXd Vec2Mat(VectorXd vec, int height, int width, const std::string output_image_path);
SparseMatrix<double,RowMajor> SharpeningMatrix(int h, int w);
SparseMatrix<double,RowMajor> EdgeMatrix(int h, int w);
VectorXd SolveSystemCG(int h, int w, SparseMatrix<double,RowMajor> mat, VectorXd x);

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
      img(i, j) = static_cast<double>(image_data[index]); 
    }
  }

  // Free memory
  stbi_image_free(image_data);

  //Task 1 We are reporting the size of the matrix
  cout <<"\nNumber of ROWS : "<<height<< endl;
  cout <<"\nNumber of COLS : "<<width<< endl;
  //Task 2
  MatrixXd img_noised = Noise(img,height,width);
  //Task 3
  VectorXd v = Mat2Vec(img,height,width);
  VectorXd w = Mat2Vec(img_noised,height,width);
  cout<< "\nMatrix elements : "<< height*width << "   v elements : "<< v.size()<< "  w elements : "<< w.size()<<endl;
  cout<< "\nNorm of the vector v is: "<< v.norm()<<endl;
  //Task 4
  SparseMatrix<double,RowMajor> A1 = SmoothingMatrix(height,width);
  VectorXd multipl = filter_grey(A1*v);
  //Task 5
  VectorXd filtered_img = filter_grey(A1*w);
  Vec2Mat(filtered_img, height,width,"output_smoothed.png");
  //Task 6
  SparseMatrix<double,RowMajor> A2 = SharpeningMatrix(height,width);
  VectorXd sharpened_img = filter_grey(A2*v);
  isSparseSymmetric(A2,"A2");
  //Task 7
  Vec2Mat(sharpened_img, height,width,"output_sharpened.png");
  //Task 8
  //Task 9
  //Task 10
  SparseMatrix<double,RowMajor> A3 = EdgeMatrix(height,width);
  VectorXd edge_img = filter_grey(A3*v);
  isSparseSymmetric(A3,"A3");
  //Task 11
  Vec2Mat(edge_img, height,width,"output_laplaceEdge.png");
  //Task 12
  VectorXd sys_sol = filter_grey(SolveSystemCG(height,width,A3,w));
  //Task 13
  Vec2Mat(sys_sol,height,width,"output_sysolution.png");
  

  return 0;
}

//Filter values > 255 and values < 0
VectorXd filter_grey(VectorXd v){
  for(int i = 0; i < v.size() ; i++){
    if( v(i) > 255 ) v(i) = 255;
    if( v(i) < 0 ) v(i) =0;
  }
  return v;
}

//Create the convolution Matrix
SparseMatrix<double,RowMajor> createConvolution(std::vector<double> values, int height, int width ){
    int dim = height * width;
    SparseMatrix<double,RowMajor> A(dim,dim);
    std::vector<T> tripletList;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {  
            int pixelIndex = i * width + j;
            for (int ki = 0; ki < 3; ++ki) {
                for (int kj = 0; kj < 3; ++kj) {
                    int ni = i + ki - 1;  
                    int nj = j + kj - 1;  
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int neighborIndex = ni * width + nj; 
                        if(values[3*ki + kj] != 0) tripletList.push_back(T(pixelIndex, neighborIndex, values[3*ki + kj]));  
                    }
                }
            }
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}

//Check if a Sparse matrix is Symmetric
void isSparseSymmetric(SparseMatrix<double,RowMajor> A2, const std::string mn){
  SparseMatrix<double,RowMajor> A2_T = A2.transpose();
  SparseMatrix<double,RowMajor>  A2_diff = A2_T - A2;
  if( A2_diff.norm() != 0 ) cout<<"\nMatrix "<<mn<<" is NOT Symmetric"<<endl;
  else cout<<"\nMatrix "<<mn<<" IS Symmetric"<<endl;
}

//
bool checkFactorisation(SparseMatrix<double,RowMajor> mat){
 return false;
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
  const std::string output_image_path = "./OUTPUT/output_noised.png";
  if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                     grayscale_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
  }
    std::cout << "\nImage saved to " << output_image_path << std::endl;

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
  double v = 1.0/9.0;
    vector<double> values = {v, v , v, v, v , v, v, v , v};
  SparseMatrix<double,RowMajor> mat = createConvolution(values, h , w);
  int nz = mat.nonZeros();
  cout<< "\nNumber of Non Zeros A1 : "<< nz << " --->  " << 100*(double(nz)/double(mat.size()))<< "% filled"<<endl;
  //cout<< "\n"<< MatrixXd(mat)<<endl;
  return mat;
}


//Task N.5/7 Apply the previous smoothing filter to the noisy image by performing the matrix vector
//multiplication A1w. Export and upload the resulting image.
MatrixXd Vec2Mat(VectorXd vec, int height, int width,const std::string output_image_path){

   // Create the "OUTPUT" directory if it doesn't exist
    std::string output_dir = "OUTPUT";

    // Update the output path to include the "OUTPUT" folder
    std::string full_output_path = output_dir + "/" + output_image_path ;

    MatrixXd mat(height,width);
    int index = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
        index = i * width + j;
        mat(i,j) = vec[index];
        }
    }

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
  grayscale_image = mat.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });

  // Save the grayscale image using stb_image_write
  if (stbi_write_png(full_output_path.c_str(), width, height, 1,
                     grayscale_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
  }
    std::cout << "\nImage saved to " << output_image_path << std::endl;

    
  return mat;
}


//Task N.6 Write the convolution operation corresponding to the sharpening kernel Hsh2 as a matrix
//vector multiplication by a matrix A2 having size mn × mn. Report the number of non-zero
//entries in A2. Is A2 symmetric?
SparseMatrix<double,RowMajor> SharpeningMatrix(int h, int w){
  vector<double> values = {0.0, -3.0 , 0.0, -1.0, 9.0 , -3.0, 0.0, -1.0 , 0.0};
  SparseMatrix<double,RowMajor> mat = createConvolution(values, h , w);
  int nz = mat.nonZeros();
  cout<< "\nNumber of Non Zeros A2 : "<< nz << " --->  " << 100*(double(nz)/double(mat.size()))<< "% filled"<<endl;
  //cout<< "\n"<< MatrixXd(mat)<<endl;
  return mat;
}


//Task N.10 Write the convolution operation corresponding to the detection kernel Hlap as a matrix
//vector multiplication by a matrix A3 having size mn × mn. Is matrix A3 symmetric?
SparseMatrix<double,RowMajor> EdgeMatrix(int h, int w){
  vector<double> values = {0.0, -1.0 , 0.0, -1.0, 4.0 , -1.0, 0.0, -1.0 , 0.0};
  SparseMatrix<double,RowMajor> mat = createConvolution(values, h , w);
  int nz = mat.nonZeros();
  cout<< "\nNumber of Non Zeros A1 : "<< nz << " --->  " << 100*(double(nz)/double(mat.size()))<< "% filled"<<endl;
  //cout<< "\n"<< MatrixXd(mat)<<endl;
  return mat;
}


VectorXd SolveSystemCG(int h, int w, SparseMatrix<double,RowMajor> mat, SpVec b){
  for ( int i = 0; i<h*w ; i++ ){ mat.coeffRef(i,i) += 1.0; }
  SpVec x(h*w);
  // Set parameters for solver
  double tol = 1.e-10;                 // Convergence tolerance
  int result, maxit = 1000;           // Maximum iterations
  // Solving 
  Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
  cg.setMaxIterations(maxit);
  cg.setTolerance(tol); 
  cg.compute(mat);  //factorization of the matrix
  //It's crucial to check that CG is feasible (so mat has to be SDP)
  if( cg.info() == Eigen::Success ){
    
    x = cg.solve(b); //the actual compution of the solution is done by the fun solve
    cout << "\n#iterations:     " << cg.iterations() << endl;
    cout << "\nrelative residual: " << cg.error()  << endl;
  }
  else{
    cout<< " CG non app "<<endl;
  }

  return x;
}