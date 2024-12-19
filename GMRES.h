#ifndef GMRES_H
#define GMRES_H

#include"Input_Data.h"
#include "Generate_Random.h"

template<typename MatrixType_, typename VectorType_, typename VectorCType_>
void Pre_GMRES(const MatrixType_& mass, const MatrixType_& stiff, VectorType_& Uk, const VectorType_& b_k, VectorType_& Da,
	 const VectorCType_& A, const VectorCType_& B, int row, int Nt, int max_iter, fp64 dt, fp64 tol)
{
	VectorTyped x_0 = VectorTyped::Zero(row * Nt);
	VectorTyped res = VectorTyped::Zero(row * Nt);
	VectorTyped ku = VectorTyped::Zero(row * Nt);
	//计算初始残差
	K_times_U(ku, x_0, mass, stiff,  dt, row, Nt); //计算K*u  高精度计算

	VectorType_ bAx0 = b_k - ku; //(b - Ax0) 这里的b_k 应该是外层循环计算的r_k
	invP_times_U(res, bAx0, Da, mass, stiff, A, B, row, Nt, dt); //r=M^{-1}(b - Ax0) 低精度计算用于求解step-b中的方程组
	std::vector<VectorType_> q_k;
	fp64 beta = res.norm();
	q_k.emplace_back(res / beta);
	VectorType_ w_k = VectorType_::Zero(Nt * row);
	MatrixType_ Hess = MatrixType_::Zero(max_iter + 1, max_iter);

	//计算w = M^{-1}Aq_k

	for (int k = 0; k < max_iter; ++k)
	{
		//MatrixType_ Hess = MatrixType_::Zero(k + 2, k + 1);
		VectorType_ Aq = VectorType_::Zero(Nt * row);
		VectorType_ beta_e = VectorType_::Zero(k + 2);
		beta_e(0) = beta;
		//计算w=M^{-1}Aq_k
		K_times_U(Aq, q_k[k], mass, stiff,  dt, row, Nt); //计算K*u  高精度计算
		invP_times_U(w_k, Aq, Da, mass, stiff, A, B, row, Nt, dt);
		for (int j = 0; j <= k; ++j)
		{
			Hess(j, k) = w_k.transpose() * q_k[j]; //h_{jk} = w^{T}q_j
			w_k = w_k - Hess(j, k) * q_k[j]; // w = w - h_{jk}q_j
		}
		Hess(k + 1, k) = w_k.norm(); //h_{k+1,k} = ||w||_2
		q_k.emplace_back(w_k / Hess(k + 1, k));
		MatrixType_ subH = Hess.block(0, 0, k + 2, k + 1);
		//todu 求解最小二乘问题
		VectorType_ c_k(k + 1);
		c_k = subH.fullPivHouseholderQr().solve(beta_e);

		VectorType_ g_res = beta_e - subH * c_k;
		printf("Error for %d iteration of GMRES is %2.15f\n", k, g_res.norm());
		if (g_res.norm() < tol)
		{
			//计算x_k = Q_k*c_k + x_0
			MatrixType_ Q_k(Nt * row, k + 1);
			for (int j = 0; j <= k; ++j)
			{
				Q_k.col(j) = q_k[j];
			}
			Uk = x_0 + Q_k * c_k;
			break;
		}
	}
}



#endif // !__GMRES_H__

