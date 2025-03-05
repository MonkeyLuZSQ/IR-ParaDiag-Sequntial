#ifndef GENERATE_RANDOM_H
#define GENERATE_RANDOM_H
#include"Input_Data.h"
#include<random>
#include<fftw3.h> //头文件用于傅里叶变换操作

#define PI acos(-1)

using fp32 = float; // single-precision
using fp64 = double; //double-precision

template<typename VectorType_>
inline void generate_random(int row, int Nt, VectorType_& F)
{
	//生成随机数
	std::random_device rd; //种子源，初始化std::mt19937生成器
	std::mt19937 mt(rd()); //STL: Mersenne twister 快速生成高质量随机数
	std::uniform_real_distribution<fp64> dist(0, 2);

	int total = row * Nt;
	for (int i = 0; i < total; ++i)
	{
		F(i) = dist(mt);
	}
}

//计算参考解
template<typename MatrixType1_, typename MatrixType2_, typename VectorType_>
inline void calculate_d_ref(MatrixType1_& d_ref, const MatrixType2_& mass, const MatrixType2_& stiff, const VectorType_& F, const VectorType_& d0, const VectorType_& dt0, int row, int Nt, fp64 dt)
{
	VectorType_ d1(row);
	fp64 cons = 0.5 * pow(dt, 2);
	d1 = d0 + dt * dt0 + cons * (mass.inverse() * (-stiff * d0 + F));
	d_ref.col(0) = d0.template cast<fp64>();
	d_ref.col(1) = d1.template cast<fp64>();
	MatrixType2_ MK, inv_MK, inv_MK1;
	MK = mass + cons * stiff;
	inv_MK = MK.inverse();
	/*for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < row; ++j)
			for (int k = 0; k < row; ++k)
				inv_MK1(i, k) += 2 * inv_MK(i, j) * mass(j, k);
	}*/
	inv_MK1 = 2 * inv_MK * mass;
	for (int i = 2; i < Nt + 2; ++i)
	{
		VectorTyped veci1 = d_ref.col(i - 1), veci2 = d_ref.col(i - 2);

		d_ref.col(i) = inv_MK1.template cast<fp64>() * veci1 - veci2 + pow(dt, 2) * inv_MK.template cast<fp64>() * F.template cast<fp64>();
	}
}


template<typename MatrixType_h, typename MatrixType_l, typename VectorType_l = VectorTypef, typename MatrixCType_, typename VectorCType_>
inline void calculate_step_b(const MatrixType_h& mass, const MatrixType_h& stiff, MatrixType_l& X_R, MatrixType_l& X_I, const MatrixCType_& B_n, const VectorCType_& A, const VectorCType_& B, VectorCType_& d_t, int mrank, int row, int col, fp64 dt)
{
	MatrixType_l Q_R(row, col), Q_I(row, col); //
	VectorType_l B_R(row), B_I(row);
	fp32 cons = 0.5 * pow(dt, 2);
	for (int j = 0; j < row; ++j)
	{
		for (int k = 0; k < col; ++k)
		{
			fp32 re = static_cast<fp32>(mass(j, k)) * A.real()(mrank) + cons * static_cast<fp32>(stiff(j, k)) * B.real()(mrank); //实部
			fp32 im = static_cast<fp32>(mass(j, k)) * A.imag()(mrank) + cons * static_cast<fp32>(stiff(j, k)) * B.imag()(mrank); //虚部
			Q_R(j, k) = re;
			Q_I(j, k) = im;
		}
		B_R(j) = B_n(j, mrank).real();
		B_I(j) = B_n(j, mrank).imag();
	}
	//step 1: QRXR=BR+QIXI -> XR
	X_R.col(mrank) = Q_R.lu().solve(B_R + Q_I*X_I.col(mrank));
	//step 2: QRXI=BI-QXRXI -> XI
	X_I.col(mrank) = Q_I.lu().solve(B_I - Q_R*X_R.col(mrank));
	for (int j = 0; j < row; ++j)
	{
		d_t(j) = std::complex<fp32>(X_R(j, mrank), X_I(j, mrank));
	}
}


template<typename VectorType_, typename MatrixCType_>
inline void FFTW(const VectorType_& res, const VectorType_& Da, MatrixCType_& S1, int row, int Nt)
{
	/*将对矩阵的傅里叶变换改为对矩阵的每一列进行傅里叶变换*/
	for (int i = 0; i < row; ++i)
	{
		fftw_complex* in_t = new fftw_complex[Nt];
		fftw_complex* out_t = new fftw_complex[Nt];

		fftw_plan P;
		P = fftw_plan_dft_1d(Nt, in_t, out_t, FFTW_FORWARD, FFTW_ESTIMATE);
		for (int j = 0; j < Nt; ++j)
		{
			in_t[j][0] = static_cast<fp32>(Da(j) * res(j * row + i));
			in_t[j][1] = 0;
		}

		fftw_execute(P);

		for (int j = 0; j < Nt; ++j)
		{
			S1(i, j) = std::complex<fp32>(out_t[j][0], out_t[j][1]);
		}

		fftw_destroy_plan(P);
		delete in_t;
		delete out_t;
	}
}

template<typename VectorType_, typename VectorCType_>
inline void inv_FFTW(const VectorCType_& D_b, VectorType_& fft_out, int Nt, int row)
{
	//先做逆傅里叶变换
	for (int i = 0; i < row; ++i)
	{
		fftw_complex* in_t = new fftw_complex[Nt];
		fftw_complex* out_t = new fftw_complex[Nt];

		fftw_plan P;
		P = fftw_plan_dft_1d(Nt, in_t, out_t, FFTW_BACKWARD, FFTW_ESTIMATE);
		for (int j = 0; j < Nt; ++j)
		{
			in_t[j][0] = (fp64)D_b(j * row + i).real();
			in_t[j][1] = (fp64)D_b(j * row + i).imag();
		}

		fftw_execute(P);

		for (int j = 0; j < Nt; ++j)
		{
			fp64 index1 = -j / (fp64)Nt;
			//std::cout << "out_t " << j << " = " << out_t[j][0] / (fp64)Nt << std::endl;
			//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			fft_out(j * row + i) = out_t[j][0] / (fp64)Nt;
		}
	}
}

template<typename MatrixType_, typename VectorType_>
inline void K_times_U(VectorType_& ku, const VectorType_& D_t, const MatrixType_& mass, const MatrixType_& stiff,  fp64 dt, int row, int Nt)
{
	fp64 cons = 0.5 * pow(dt, 2);
	for (int i = 0; i < Nt; ++i) //计算\mathcal{K}*d
	{
		if (i == 0)
		{
			ku.segment(i * row, row) = (mass.template cast<fp64>() + cons * stiff.template cast<fp64>()) * D_t.segment(i * row, row);
		}
		else if (i == 1)
		{
			ku.segment(i * row, row) = (mass.template cast<fp64>() + cons * stiff.template cast<fp64>()) * D_t.segment(i * row, row) - 2 * mass.template cast<fp64>() * D_t.segment(0, row);
		}
		else
		{
			ku.segment(i * row, row) = (mass.template cast<fp64>() + cons * stiff.template cast<fp64>()) * (D_t.segment(i * row, row) + D_t.segment((i - 2) * row, row)) - 2 * mass.template cast<fp64>() * D_t.segment((i - 1) * row, row);
		}
	}
}

template<typename VectorType_h, typename MatrixType_h, typename MatrixType_l, typename VectorCType_>
inline void invP_times_U(VectorType_h& d_Uk, const VectorType_h& res, VectorType_h& Da, const MatrixType_h& mass, 
	const MatrixType_h& stiff, MatrixType_l& X_R, MatrixType_l& X_I, const VectorCType_& A, const VectorCType_& B, int row, int Nt, fp64 dt)
{
	MatrixCTypef S1;
	VectorCType_ d_t(row); // 每个进程任务里计算step-b得到的解向量，gatherv到根进程里，
	VectorCType_ D_b = VectorCType_::Zero(row * Nt); //收集step-b求解得到的向量
	S1.resize(row, Nt);
	//正向傅里叶变换
	FFTW(res, Da, S1, row, Nt);

	auto begin_t = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < Nt; ++i)
	{
		std::cout << "			Nt = " << i << " : ================= " << i * 100 / Nt << " %done. " << std::endl;
		calculate_step_b(mass, stiff, X_R, X_I, S1, A, B, d_t, i, row, row, dt);
		for (int j = 0; j < row; ++j)
		{
			D_b(i * row + j) = d_t(j);
		}
	}

	auto end_t = std::chrono::high_resolution_clock::now();
	//std::cout << "(^_^)---------------------------Parallel computing part---------------------------(^_^)" << std::endl;
	auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin_t).count();
	std::cout << "(^_^) Solve complex linear equations successfully (^_^), and the time is " << cost << "ms. " << std::endl;
	std::cout << std::endl;
	/*更新D_t并广播给所有的进程*/
	//逆傅里叶变换
	inv_FFTW(D_b, d_Uk, Nt, row);
}


template<typename MatrixType_h, typename VectorType_h>
inline void calculate_b_k(VectorType_h& b_k, const MatrixType_h& stiff, const MatrixType_h& mass, const VectorType_h& F, const VectorType_h& d0, const VectorType_h& dt0, fp64 dt, int row, int Nt)
{
	VectorType_h d1(row);
	fp64 cons = 0.5 * pow(dt, 2);
	d1 = d0 + dt * dt0 + cons * (mass.inverse() * (-stiff * d0 + F));
	if (Nt == 1)
	{
		b_k.segment(0, row) = 2 * cons * F + 2 * mass * d1 - (mass + cons * stiff) * d0;
	}
	else if (Nt == 2)
	{
		b_k.segment(0, row) = 2 * cons * F + 2 * mass * d1 - (mass + cons * stiff) * d0;
		b_k.segment(row, row) = 2 * cons * F - (mass + cons * stiff) * d1;
	}
	else
	{
		b_k.segment(0, row) = 2 * cons * F + 2 * mass * d1 - (mass + cons * stiff) * d0;
		b_k.segment(row, row) = 2 * cons * F - (mass + cons * stiff) * d1;

		for (int i = 2; i < Nt; ++i)
		{
			b_k.segment(i * row, row) = 2 * cons * F;
		}
	}

	/* debug 输出 */
	//for (int i = 0; i < Nt * row; ++i)
	//{
	//	std::cout << "b_k: " << b_k(i) << std::endl;
	//}
}

template<typename VectorCType_, typename VectorType_>
inline void update_D_t(const VectorCType_& D_b, VectorType_& D_t, int Nt, int row, fp64 alp)
{
	//先做逆傅里叶变换
	for (int i = 0; i < row; ++i)
	{
		fftw_complex* in_t = new fftw_complex[Nt];
		fftw_complex* out_t = new fftw_complex[Nt];

		fftw_plan P;
		P = fftw_plan_dft_1d(Nt, in_t, out_t, FFTW_BACKWARD, FFTW_ESTIMATE);
		for (int j = 0; j < Nt; ++j)
		{
			in_t[j][0] = D_b(j * row + i).real();
			in_t[j][1] = D_b(j * row + i).imag();
		}

		fftw_execute(P);

		for (int j = 0; j < Nt; ++j)
		{
			fp64 index1 = -j / (fp64)Nt;
			//std::cout << "out_t " << j << " = " << out_t[j][0] / (fp64)Nt << std::endl;
			//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			D_t(j * row + i) = D_t(j * row + i) + pow(alp, index1) * (fp64)out_t[j][0] / (fp64)Nt;
		}
		//std::cout << std::endl;

		fftw_destroy_plan(P);
		delete in_t;
		delete out_t;

	}
}


template<typename VectorCType_>
inline void generate_fft(fp64 alpha, fp64 dt, int Nt, VectorCType_& A, VectorCType_& B)
{
	MatrixTypef C1(Nt, Nt), C2(Nt, Nt);
	MatrixTypef Lm(Nt, Nt);
	MatrixCTypef Ft(Nt, Nt);
	std::complex<fp64> theta(cos(2 * PI / Nt), sin(2 * PI / Nt));

	/*初始化Lambda和FFT矩阵*/
	for (int i = 0; i < Nt; ++i)
	{
		fp64 exp = i / (fp64)Nt;
		Lm(i, i) = pow(alpha, exp);
		for (int j = 0; j < Nt; ++j)
		{
			Ft(i, j) = pow(theta, i * j);
		}
	}

	/*std::cout << "Lambda: " << std::endl;
	std::cout << Lm << std::endl;

	std::cout << "FFT: " << std::endl;
	std::cout << Ft << std::endl;*/

	/*初始化C1和C2托普利兹矩阵*/
	// 填充对角线元素和次对角线元素
	for (int i = 0; i < Nt; ++i) {
		C1(i, i) = 1;
		C2(i, i) = pow(dt, 2) / 2;
		if (i > 1) {
			C1(i, i - 1) = -2;
			C1(i, i - 2) = 1;
			C2(i, i - 2) = pow(dt, 2);
		}
	}
	C1(1, 0) = -2;
	C1(0, Nt - 1) = -2 * alpha;
	C1(0, Nt - 2) = alpha;
	C1(1, Nt - 1) = alpha;
	C2(0, Nt - 2) = alpha * pow(dt, 2) / 2.0;
	C2(1, Nt - 1) = alpha * pow(dt, 2) / 2.0;

	/*std::cout << "C1: " << std::endl;
	std::cout << C1 << std::endl;

	std::cout << "C2: " << std::endl;
	std::cout << C2 << std::endl;*/

	VectorTypef C11 = C1.col(0);
	VectorTypef C21 = C2.col(0);

	A = Ft * Lm * C11;
	B = Ft * Lm * C21;

	/*std::cout << "A: " << std::endl;
	std::cout << A << std::endl;

	std::cout << "B: " << std::endl;
	std::cout << B << std::endl;

	std::cout << std::endl;*/

}

template<typename VectorType_>
void export_solution(std::string filename, VectorType_ D_t, int row, int Nt)
{
	std::ofstream file;
	file.open(filename.c_str());
	if (!file.is_open())
		std::cout << "File is not open, please check! " << std::endl;
	for (int i = 0; i < row * Nt; ++i)
	{
		file << D_t(i) << std::endl;
		//std::cout << X_t(i).real() << " " << X_t(i).imag() << std::endl;
	}
	file.close();
}


#endif // !GENERATE_RANDOM_H

