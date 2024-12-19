#ifndef INPUT_DATA_H
#define INPUT_DATA_H
#include<iostream>
#include<string>
#include<sstream>
#include<fstream>
#include<vector>
#include<map>
#include<chrono>
#include<Eigen/Dense>
#include<Eigen/Core>


#define CHUNKSIZE 4
#define FILE_K "./Input/Stiff.txt"
#define FILE_M "./Input/Mass.txt"
#define FILE_A_B "./Input/a_b.txt"
#define FILE_R "./Export/X_t.txt"
#define FILE_D "./Input/d0.txt"
#define FILE_DT "./Input/dt0.txt"
#define FILE_F "./Input/F.txt"
#define FILE_D_REF "./Export/d_ref.txt"

typedef Eigen::MatrixXd MatrixTyped;
typedef Eigen::VectorXd VectorTyped; //定义实矩阵和向量的精度格式
typedef Eigen::MatrixXcd MatrixCTyped;
typedef Eigen::VectorXcd VectorCTyped; //定义复数矩阵和向量的精度格式
typedef Eigen::MatrixXf MatrixTypef;
typedef Eigen::VectorXf VectorTypef; //定义实矩阵和向量的精度格式
typedef Eigen::MatrixXcf MatrixCTypef;
typedef Eigen::VectorXcf VectorCTypef; //定义复数矩阵和向量的精度格式
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HalfMatrix;

template<typename MatrixType_, typename VectorType_, typename VectorCType_>
class Input_Data
{
public:
	typedef Eigen::ComplexEigenSolver<MatrixType_> ces;
	//Input_Data();
	void read_matrix(std::string filename, MatrixType_& Matrix, int row, int col);
	void read_vector(std::string filename, VectorType_& vec, int row);
	void read_coefficient_A_B(std::string filename, VectorCType_& A, VectorCType_& B, int row);
	//void export_solution(std::string filename, VectorType_ X_t, int row, int Nt);
};

template<typename MatrixType_, typename VectorType_, typename VectorCtype_>  // Eigen::MatrixXd or Eigen::MatrixXf
void  Input_Data<MatrixType_, VectorType_, VectorCtype_>::read_matrix(std::string filename, MatrixType_& Matrix, int row, int col)
{
	std::ifstream infile;
	infile.open(filename.c_str());
	if (infile.is_open())
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				infile >> Matrix(i, j); // 从文件中读取矩阵元素
			}
		}
		infile.close();
	}
	else
	{
		std::cerr << "Unable to open file " << filename << std::endl;
	}
	infile.close();
	std::cout << "Read matrix data successfully " << std::endl;
}

template<typename MatrixType_, typename VectorType_, typename VectorCType_>  // Eigen::VectorXd or Eigen::VectorXf
void Input_Data<MatrixType_, VectorType_, VectorCType_>::read_vector(std::string filename, VectorType_& vec, int row)
{
	std::ifstream infile;
	infile.open(filename.c_str());
	if (infile.is_open())
	{
		for (int i = 0; i < row; ++i)
		{
			infile >> vec(i); // 从文件中读取矩阵元素
		}
		infile.close();
		std::cout << "Read vector data successfully " << std::endl;
	}
	else
	{
		std::cerr << "Unable to open file " << filename << std::endl;
	}
}

template<typename MatrixType_, typename VectorType_, typename VectorCType_> //double or float
void Input_Data<MatrixType_, VectorType_, VectorCType_>::read_coefficient_A_B(std::string filename, VectorCType_& A, VectorCType_& B, int row)
{
	std::ifstream infile;
	infile.open(filename.c_str());
	if (infile.is_open())
	{
		for (int i = 0; i < row; ++i)
		{
			double a_r, a_i, b_r, b_i;
			infile >> a_r >> a_i >> b_r >> b_i; // 从文件中读取矩阵元素 a_r a的实部，a_i a的虚部，b类似
			A(i).real(a_r), A(i).imag(a_i);
			B(i).real(b_r), B(i).imag(b_i);
		}
		infile.close();
		std::cout << "Read vector data successfully " << std::endl;
	}
	else
	{
		std::cerr << "Unable to open file " << filename << std::endl;
	}
}

//template<typename MatrixType_, typename VectorType_, typename VectorCType_>
//void Input_Data<MatrixType_, VectorType_, VectorCType_>::export_solution(std::string filename, VectorType_ D_t, int row, int Nt)
//{
//	std::ofstream file;
//	file.open(filename.c_str());
//	if (!file.is_open())
//		std::cout << "File is not open, please check! " << std::endl;
//	for (int i = 0; i < row * Nt; ++i)
//	{
//		file << D_t(i) << std::endl;
//		//std::cout << X_t(i).real() << " " << X_t(i).imag() << std::endl;
//	}
//	file.close();
//}
#endif // !INPUT_DATA_H
