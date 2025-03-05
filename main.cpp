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

	//int row = 578, col = 578, Nt_tol, Nt, tmax = 10; //����ά��(row, col)�Լ�ʱ�䲽����(Nt)
	int row = 882, col = 882, Nt_tol, Nt, tmax = 10; //����ά��(row, col)�Լ�ʱ�䲽����(Nt)

	//�ı�Nt=16, 20, 32, ...  ���ٱȺͺ����Ĺ�ϵ������Ч�ʣ�����Nt��ǰ������ֵ����֪�������ֻ�����ӵ�2����ʼ�Ľ��
	//�ռ��ʷ�Ҳ�༸�飬���Ի�����


	fp64 dt = 0.05; //ʱ�䲽
	fp64 tol = 1e-6; //���������
	fp64 alpha = 1e-3; // a parameter used in the ParaDiag algorithm
	bool flag = true; //������ֹParaDiag����

	fp64 gtol = 1e-8; //GMRES�����������
	int gmax_iter = 100; //���Krylov�ӿռ��ά�ȣ�GMRES������������
	Nt = 8;
	Nt_tol = Nt + 2;

	Input_Data<MatrixTyped, VectorTyped, VectorCTypef> Input; //����Input��
	MatrixTyped stiff, mass; // M, K
	MatrixCTypef B_n; /*�����B_nӦ����step-a�е�y^{\alpha}�����㷽ʽ�ο�����ʦ����88��S1*/
	VectorTyped d0(row), dt0(row); //��ʼ����λ�ƣ���ʼ�����ٶȣ�̩��չʽ����t=1ʱ�ĳ�ʼ����
	VectorCTypef A, B;
	VectorTyped F = Eigen::VectorXd::Zero(row); //�Ҷ��غ�����
	VectorTyped b_k = VectorTyped::Zero(row * Nt); //��ʽ5.10���Ҷ���
	VectorTyped ku = VectorTyped::Zero(row * Nt);
	VectorTyped res = VectorTyped::Zero(row * Nt); // ����в�
	stiff.resize(row, col);
	mass.resize(row, col);
	B_n.resize(row, Nt);
	A.resize(Nt);
	B.resize(Nt);
	//VectorCType X_t(row*Nt); //������
	MatrixTyped d_ref(row, Nt_tol); // �ο��⣬
	VectorTyped D_t = VectorTyped::Zero(row * Nt); // ��ǰʱ�䲽�Ľ�������ʵ��
	MatrixTypef X_R = MatrixTypef::Zero(row, Nt); // ���ڴ洢ÿһ������ʵ��������
	MatrixTypef X_I = MatrixTypef::Zero(row, Nt); // ���ڴ洢ÿһ�������鲿�����ݣ�������������
	VectorTyped Err = VectorTyped::Zero(tmax + 1);

	//��ȡ����
	std::cout << "(-_-)---------------------------Pre-processing data part---------------------------(-_-)" << std::endl;
	auto begin_t = std::chrono::high_resolution_clock::now();
	Input.read_matrix(FILE_K, stiff, row, col);
	Input.read_matrix(FILE_M, mass, row, col);
	Input.read_vector(FILE_D, d0, row);
	Input.read_vector(FILE_DT, dt0, row);
	Input.read_vector(FILE_F, F, row);
	calculate_b_k(b_k, stiff, mass, F, d0, dt0, dt, row, Nt); //����bk�Ҷ���  ��ʽ5.10���Ҷ����Ӧ����ʦ������64-69 vecb

	auto end_t = std::chrono::high_resolution_clock::now();
	uint64_t cost = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin_t).count();
	std::cout << "Read data successfully (^_^), and the time is " << cost << " ms. " << std::endl;
	std::cout << std::endl;
	//����ο���
	calculate_d_ref(d_ref, mass, stiff, F, d0, dt0, row, Nt, dt);
	d_ref.resize(row * Nt_tol, 1);
	export_solution(FILE_D_REF, d_ref, row, Nt_tol);
	Err(0) = (D_t.segment((Nt - 2) * row, row) - d_ref.col(0).segment((Nt_tol - 2) * row, row)).norm();
	printf("The initial ParaDiag_Error at is % 2.15f\n", Err(0));
	//generate_random(row, Nt, D_t); //������ֵ

	Input.read_coefficient_A_B(FILE_A_B, A, B, Nt);
	VectorTyped Da(Nt);
	for (int i = 0; i < Nt; ++i)
	{
		fp64 index1 = i / (fp64)Nt;
		Da(i) = pow(alpha, index1);
	}

	/*ʱ�䲽���� t = 0, 1, 2, ..., tmax*/
	/* ���н��̶�ִ�д��ѭ������step-b��ʱ��������񣬽��������ռ�������̣�������������rk��������һ���ĵ�������ʼ��һ������ */

	/*��ʼ��ʱ zsq1*/
	auto begin_T = std::chrono::high_resolution_clock::now();
#if IS_GMRES  //GMRES IR-ParaDiag
	for (int t = 0; t < tmax; ++t)
	{
		if (!flag) break;
		VectorTyped d_Uk = VectorTyped::Zero(row * Nt); // ��ǰʱ�䲽��������ʵ��
		// ����rk
		K_times_U(ku, D_t, mass, stiff, dt, row, Nt); //����K*u  �߾��ȼ���
		res = b_k - ku;
		std::cout << "(-_-)---------------------------" << t + 1 << "'s  Parallel computing part begin -------------------------- - (-_-)" << std::endl;
		Pre_GMRES(mass, stiff, d_Uk, res, Da, X_R, X_I, A, B, row, Nt, gmax_iter, dt, gtol);

		//����D_t
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
#else //������ IR ParaDiag
	VectorTyped d_Uk = VectorTyped::Zero(row * Nt);
	for (int t = 0; t < tmax; ++t)
	{
		if (!flag) break;

		K_times_U(ku, D_t, mass, stiff, dt, row, Nt); //����K*u
		res = b_k - ku; //(b - Ax0);
		invP_times_U( d_Uk, res, Da, mass, stiff, X_R, X_I, A, B, row, Nt, dt); 

		//����D_t
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

	/*���������������*/
	std::cout << "(-_-)---------------------------Result output part---------------------------(-_-)" << std::endl;
	auto begin_t2 = std::chrono::high_resolution_clock::now();
	export_solution(FILE_R, D_t, row, Nt);
	auto end_t2 = std::chrono::high_resolution_clock::now();
	auto cost2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_t2 - begin_t2).count();
	std::cout << "(^_^) The output file was successfully written (^_^), and the time is " << cost2 << " ms. " << std::endl;

	return 0;
}

