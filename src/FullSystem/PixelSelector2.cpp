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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

	currentPotential=3;


	gradHist = new int[100*(1+w/32)*(1+h/32)];//不理解为什么乘以100
	ths = new float[(w/32)*(h/32)+100];//不理解为什么+100
	thsSmoothed = new float[(w/32)*(h/32)+100];//同上

	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

int computeHistQuantil(int* hist, float below)
{
	int th = hist[0]*below+0.5f;//hist[0]为统计的点个数，below为0.5，th为点个数的一半
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return 90;
}


void PixelSelector::makeHists(const FrameHessian* const fh)
{
	gradHistFrame = fh;
	float * mapmax0 = fh->absSquaredGrad[0];

	int w = wG[0];
	int h = hG[0];

	int w32 = w/32;
	int h32 = h/32;
	thsStep = w32;

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float* map0 = mapmax0+32*x+32*y*w;//把这块32x32的区域拿出来放到map0里
			int* hist0 = gradHist;// + 50*(x+y*w32);
			memset(hist0,0,sizeof(int)*50);

			for(int j=0;j<32;j++) for(int i=0;i<32;i++)
			{
				int it = i+32*x;
				int jt = j+32*y;
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
				int g = sqrtf(map0[i+j*w]);
				if(g>48) g=48;
				hist0[g+1]++;
				hist0[0]++;//hist0[0]应该等于32x32，除了那些边缘块会少一些
			}

			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
		}

	//对32*32个计算的区域阈值进行3x3邻域的均值平滑
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		}





}
int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;


//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else
	{




		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

		if(fh != gradHistFrame) makeHists(fh);//将图像分成32x32块，计算每块的梯度中值作为选点的阈值

		// select!
		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

		// sub-select!
		numHave = n[0]+n[1]+n[2];
		quotia = numWant / numHave;

		// by default we want to over-sample by 40% just to be sure.
        // 相当于覆盖的面积, 每一个像素对应一个pot*pot
		float K = numHave * (currentPotential+1) * (currentPotential+1);
        // 除以目标点数，得到应该设置的pot大小
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);//一个递归函数
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	int numHaveSub = numHave;
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));
	currentPotential = idealPotential;


	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
	}

	return numHaveSub;
}



Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{

	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];// 图像金字塔第一层梯度平方和 dx*dx + dy*dy
	float * mapmax1 = fh->absSquaredGrad[1];// 金字塔第二层梯度平方和 dx*dx + dy*dy
	float * mapmax2 = fh->absSquaredGrad[2];// 金字塔第三层梯度平方和 dx*dx + dy*dy


	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];


    //取网格内随机方向梯度最大的点
	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};

	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));



	float dw1 = setting_gradDownweightPerLevel;
	float dw2 = dw1*dw1;


	int n3=0, n2=0, n4=0;
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];//随机选择一个方向
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;
			int y34 = y3+y4;
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];
				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					int idx = x1+x234 + w*(y1+y234);
					int xf = x1+x234;
					int yf = y1+y234;

					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;


					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];//右移5位是因为之前按照32x32分块的
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;

                    //在0层上寻找梯度平方和 > thsSmoothed上对应位置的0.75倍的点，且该点在pot中最大
					float ag0 = mapmax0[idx];
					if(ag0 > pixelTH0*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();//拿到改像素的dx,dy
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));
						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					if(bestIdx3==-2) continue;

                    //如果在0层中找不到，就到金字塔1层上取找，
					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

                    //如果在1层中找不到，就到金字塔2层上取找，
					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}

				if(bestIdx2>0)
				{
					map_out[bestIdx2] = 1;
					bestVal3 = 1e10;
					n2++;
				}
			}

			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}


	return Eigen::Vector3i(n2,n3,n4);
}


}

