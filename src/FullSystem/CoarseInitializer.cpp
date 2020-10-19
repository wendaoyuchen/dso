/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    /*
     * 当检测到位移足够大时，开始从金字塔顶层向底层使用LM优化位姿，光度参数，逆深度。
     * 然后将逆深度由底层向顶层传播逆深度，用于下次优化做初值。
     * 优化到满足位移的后5帧，位移小或中间的帧删除fh
     */
	newFrame = newFrameHessian;

    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};



	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

    //snapped体现的是两帧之间的位移，如果为false，表示位移不是足够大
	if(!snapped)
	{
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}


	SE3 refToNew_current = thisToNext;
	AffLight refToNew_aff_current = thisToNext_aff;

	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.


	Vec3f latestRes = Vec3f::Zero();
    //从金字塔顶端向底层进行，体现了coarse-to-fine的思想
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{

		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1);//逆深度在不同层之间传递使用parent点来作为关联，融合策略采用高斯归一化积

		Mat88f H,Hsc; Vec8f b,bsc;//从命名上来看，H应该表示Hessian矩阵，b应该就是对应的Hx=b中的b,然后Hsc和bsc应该是经过舒尔补消元后的矩阵
		resetPoints(lvl);
        //计算Hessian矩阵,雅克比和误差计算等消息
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

        //迭代计算(H + lambda*D_T*D) * Δx = g
		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));


			Vec8f inc;//x_beta,八维，前六位是位姿，后两位是光度参数a,b
			if(fixAffine)
			{
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			}
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.


			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);//更新点的逆深度信息


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{

				if(resNew[1] == alphaK*numPoints[lvl])
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	}



	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);




	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;



    debugPlot(0,wraps);



	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;



	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Accumulator11 E;
	acc9.initialize();
	E.initialize();


	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	for(int i=0;i<npts;i++)
	{

		Pnt* point = ptsl+i;

		point->maxstep = 1e10;
		if(!point->isGood)
		{
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;
        VecNRf r;
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		float energy=0;
        //对应那个八维残差的结构
		for(int idx=0;idx<patternNum;idx++)
		{
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];


			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];//这个其实是正确的，能够理解，符合公式

			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);//在新帧上投影点的像素值，dx,dy
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);//参考帧上的像素值

			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}


			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];//E = I2(p2) - exp(a)I1(p1) - b
            //w_h = 1, if |r| < theta
            //w_h = theta/|r|, if |r| >= theta
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);//Huber范数形式
			energy += hw *residual*residual*(2-hw);//H(r) = w_h * r * r * (1 - w_h/2),这样energy就表示这八个点的总能量和



            //求导部分主要包括三个，对光度误差求导，对相对位姿求导以及对逆深度的求导，
            //https://blog.csdn.net/xxxlinttp/article/details/89379785
            // ∂f(x)/∂ϵ = sqrt(w_h) * { ▽I_x * ρ2 * f_x,
            //                          ▽I_y * ρ2 * f_y,
            ​//                          -ρ2 * (▽I_x * f_x * u_2 - ▽I_y * f_y * v_2),
            //                          -▽I_x * f_x * u_2 * v_2 - ▽I_y * f_y * (1 + v_2 * v_2),
            //                          ▽I_x * f_x * (1 + u_2 * u_2) + ▽I_y * f_y * u_2 * v_2,
            //                          -▽I_x * f_x * v_2 + ▽I_y * f_y * u_2 }
			float dxdd = (t[0]-t[2]*u)/pt[2];//(t_x - u * t_z) / pt[2] ,这里的pt[2]应该是接在t的分母的，因为u已经除以过pt[2]了
			float dydd = (t[1]-t[2]*v)/pt[2];//(t_x - v * t_z) / pt[2]

            //dp0 - dp5 是相对位姿求导的结果
            //dp6, dp7是对光度误差的求偏导
            //dd是对逆深度进行求导
			if(hw < 1) hw = sqrtf(hw);//对hw求根号，雅各比矩阵的公共系数
			float dxInterp = hw*hitColor[1]*fxl;//sqrt(w_h) * ▽I_x * f_x,
			float dyInterp = hw*hitColor[2]*fyl;//sqrt(w_h) * ▽I_y * f_y
			dp0[idx] = new_idepth*dxInterp;//▽I_x * ρ2 * f_x
			dp1[idx] = new_idepth*dyInterp;//▽I_y * ρ2 * f_y
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);//-ρ2 * (▽I_x * f_x * u_2 - ▽I_y * f_y * v_2)
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;//-▽I_x * f_x * u_2 * v_2 - ▽I_y * f_y * (1 + v_2 * v_2)
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;//▽I_x * f_x * (1 + u_2 * u_2) + ▽I_y * f_y * u_2 * v_2
			dp5[idx] = -v*dxInterp + u*dyInterp;//-▽I_x * f_x * v_2 + ▽I_y * f_y * u_2
			dp6[idx] = - hw*r2new_aff[0] * rlR;//∂f(x)/∂a
			dp7[idx] = - hw*1;//∂f(x)/∂b
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;//对逆深度的求导,
			r[idx] = hw*residual;//sqrt(w_h)* r

			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		}

		if(!isGood || energy > point->outlierTH*20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}


		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));


		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);


	}

	E.finish();
	acc9.finish();






	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
    //当位移足够大，alphaOpt = 0
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}


	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;



	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;





	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}

void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl-1];

	for(int i=0;i<nptst;i++)
	{
		Pnt* point = ptst+i;
		Pnt* parent = ptss+point->parent;

		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		if(!point->isGood)
		{
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		else
		{
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector sel(w[0],h[0]);

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1};
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);



		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl];
		int nl = 0;
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				pl[nl].u = x+0.1;
				pl[nl].v = y+0.1;
				pl[nl].idepth = 1;
				pl[nl].iR = 1;
				pl[nl].isGood=true;
				pl[nl].energy.setZero();
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
				float sumGrad2=0;
				for(int idx=0;idx<patternNum;idx++)
				{
					int dx = patternP[idx][0];
					int dy = patternP[idx][1];
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
					sumGrad2 += absgrad;
				}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//

				pl[nl].outlierTH = patternNum*setting_outlierTH;



				nl++;
				assert(nl <= npts);
			}
		}


		numPoints[lvl]=nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	//k-最近邻建立kd-tree
	//makeNN计算每个点最邻近的10个点 neighbours，在上一层的最邻近点 parent
	makeNN();

	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}

void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;


		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}




void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

