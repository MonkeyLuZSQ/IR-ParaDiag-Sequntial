//#define EIGEN_USE_MKL_ALL
//#define EIGEN_VECTORIZE_SSE4_2
#include "Generate_Random.h"
#include "GMRES.h"
#include<cmath>
//#include<mkl.h>
#define IS_GMRES 0

int main(int argc, char* argv[])
{
	system("mkdir Export");

	//int row = 578, col = 578, Nt_tol, Nt, tmax = 10; //矩阵维数(row, col)以及时间步总数(Nt)
	int row = 882, col = 882, Nt_tol, Nt, tmax = 10; //矩阵维数(row, col)以及时间步总数(Nt)

	//改变Nt=16, 20, 32, ...  加速比和核数的关系，加速效率，对于Nt，前两步的值可以知道，因此只需计算从第2步开始的结果
	//空间剖分也多几组，测试机器，


	fp64 dt = 0.05; //时间步
	fp64 tol = 1e-6; //迭代误差限
	fp64 alpha = 1e-3; // a parameter used in the ParaDiag algorithm
	bool flag = true; //用于终止ParaDiag迭代

	fp64 gtol = 1e-8; //GMRES迭代误差容限
	int gmax_iter = 100; //最大Krylov子空间的维度（GMRES最大迭代次数）
	Nt = 8;
	Nt_tol = Nt + 2;

	Input_Data<MatrixTyped, VectorTyped, VectorCTypef> Input; //声明Input类
	MatrixTyped stiff, mass; // M, K
	MatrixCTypef B_n; /*这里的B_n应该是step-a中的y^{\alpha}，计算方式参考吴老师程序88行S1*/
	VectorTyped d0(row), dt0(row); //初始条件位移，初始条件速度，泰勒展式构造t=1时的初始条件
	VectorCTypef A, B;
	VectorTyped F = Eigen::VectorXd::Zero(row); //右端载荷向量
	VectorTyped b_k = VectorTyped::Zero(row * Nt); //公式5.10的右端项
	VectorTyped ku = VectorTyped::Zero(row * Nt);
	VectorTyped res = VectorTyped::Zero(row * Nt); // 定义残差
	stiff.resize(row, col);
	mass.resize(row, col);
	B_n.resize(row, Nt);
	A.resize(Nt);
	B.resize(Nt);
	//VectorCType X_t(row*Nt); //解向量
	MatrixTyped d_ref(row, Nt_tol); // 参考解，
	VectorTyped D_t = VectorTyped::Zero(row * Nt); // 当前时间步的解向量，实数
	MatrixTypef X_R = MatrixTypef::Zero(row, Nt); // 用于存储每一步迭代实部的数据
	MatrixTypef X_I = MatrixTypef::Zero(row, Nt); // 用于存储每一步迭代虚部的数据，用以两步迭代
	VectorTyped Err = VectorTyped::Zero(tmax + 1);

	//读取数据
	std::cout << "(-_-)---------------------------Pre-processing data part---------------------------(-_-)" << std::endl;
	auto begin_t = std::chrono::high_resolution_clock::now();
	Input.read_matrix(FILE_K, stiff, row, col);
	Input.read_matrix(FILE_M, mass, row, col);
	Input.read_vector(FILE_D, d0, row);
	Input.read_vector(FILE_DT, dt0, row);
	Input.read_vector(FILE_F, F, row);
	calculate_b_k(b_k, stiff, mass, F, d0, dt0, dt, row, Nt); //计算bk右端项  公式5.10的右端项，对应吴老师程序里64-69 vecb

	auto end_t = std::chrono::high_resolution_clock::now();
	uint64_t cost = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin_t).count();
	std::cout << "Read data successfully (^_^), and the time is " << cost << " ms. " << std::endl;
	std::cout << std::endl;
	//计算参考解
	calculate_d_ref(d_ref, mass, stiff, F, d0, dt0, row, Nt, dt);
	d_ref.resize(row * Nt_tol, 1);
	export_solution(FILE_D_REF, d_ref, row, Nt_tol);
	Err(0) = (D_t.segment((Nt - 2) * row, row) - d_ref.col(0).segment((Nt_tol - 2) * row, row)).norm();
	printf("The initial ParaDiag_Error at is % 2.15f\n", Err(0));
	//generate_random(row, Nt, D_t); //迭代初值

	Input.read_coefficient_A_B(FILE_A_B, A, B, Nt);
	VectorTyped Da(Nt);
	for (int i = 0; i < Nt; ++i)
	{
		fp64 index1 = i / (fp64)Nt;
		Da(i) = pow(alpha, index1);
	}

	/*时间步迭代 t = 0, 1, 2, ..., tmax*/
	/* 所有进程都执行大的循环，在step-b的时候分配任务，将计算结果收集到零进程，在零进程里计算rk，走完这一步的迭代，开始下一步迭代 */

	/*开始计时 zsq1*/
	auto begin_T = std::chrono::high_resolution_clock::now();
#if IS_GMRES  //GMRES IR-ParaDiag
	for (int t = 0; t < tmax; ++t)
	{
		if (!flag) break;
		VectorTyped d_Uk = VectorTyped::Zero(row * Nt); // 当前时间步的增量，实数
		// 计算rk
		K_times_U(ku, D_t, mass, stiff, dt, row, Nt); //计算K*u  高精度计算
		res = b_k - ku;
		std::cout << "(-_-)---------------------------" << t + 1 << "'s  Parallel computing part begin -------------------------- - (-_-)" << std::endl;
		Pre_GMRES(mass, stiff, d_Uk, res, Da, X_R, X_I, A, B, row, Nt, gmax_iter, dt, gtol);

		//更新D_t
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < Nt; ++j)
			{
				D_t(j * row + i) = D_t(j * row + i) + /*pow(alpha, index1) * */ d_Uk(j * row + i);
			}
			//std::cout << std::endl;
		}
		Err(t + 1) = (D_t.segment((Nt - 2) * row, row) - d_ref.col(0).segment((Nt_tol - 2) * row, row)).norm();
		printf("ParaDiag_Error at % d - th iteration is % 2.15f\n", t + 1, Err(t + 1));
		std::cout << "(^_^)---------------------------" << t + 1 << "'s  Parallel computing part end --------------------------- (^_^)" << std::endl;
		std::cout << std::endl;

		if (Err(t + 1) < tol) flag = false;

	}
#else //不动点 IR ParaDiag
	VectorTyped d_Uk = VectorTyped::Zero(row * Nt);
	for (int t = 0; t < tmax; ++t)
	{
		if (!flag) break;

		K_times_U(ku, D_t, mass, stiff, dt, row, Nt); //计算K*u
		res = b_k - ku; //(b - Ax0);
		invP_times_U( d_Uk, res, Da, mass, stiff, X_R, X_I, A, B, row, Nt, dt); 

		//更新D_t
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < Nt; ++j)
			{
				fp64 index1 = -j / (fp64)Nt;
				D_t(j * row + i) = D_t(j * row + i) + pow(alpha, index1) * d_Uk(j * row + i);
			}
			//std::cout << std::endl;
		}
		Err(t + 1) = (D_t.segment((Nt - 2) * row, row) - d_ref.col(0).segment((Nt_tol - 2) * row, row)).norm();
		printf("ParaDiag_Error at % d - th iteration is % 2.15f\n", t + 1, Err(t + 1));
		std::cout << "(^_^)---------------------------" << t + 1 << "'s  Parallel computing part end --------------------------- (^_^)" << std::endl;
		std::cout << std::endl;

		if (Err(t + 1) < tol) flag = false;

	}
#endif

	
	auto end_T = std::chrono::high_resolution_clock::now();
	auto cost_T = std::chrono::duration_cast<std::chrono::milliseconds>(end_T - begin_T).count();
	std::cout << " The total ParaDiag iteration process took " << cost_T << " ms. " << std::endl;

	/*输出结果，计算误差*/
	std::cout << "(-_-)---------------------------Result output part---------------------------(-_-)" << std::endl;
	auto begin_t2 = std::chrono::high_resolution_clock::now();
	export_solution(FILE_R, D_t, row, Nt);
	auto end_t2 = std::chrono::high_resolution_clock::now();
	auto cost2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_t2 - begin_t2).count();
	std::cout << "(^_^) The output file was successfully written (^_^), and the time is " << cost2 << " ms. " << std::endl;

	return 0;
}

