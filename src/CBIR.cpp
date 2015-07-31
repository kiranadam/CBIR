/***********************************************************/
/*      Content Based Image Retrival System  C++           */
/*            Author : Kirankumar V. Adam                  */
/***********************************************************/

#include "CBIR.h"
#include <iostream>
#include <math.h>
#include <boost/range/numeric.hpp>
#include <vector>

using namespace std;
using namespace boost;

// Parametrized Constructor for CBIR class
CBIR :: CBIR(int k_size, vector<double>& sigma, vector<double>& theta, double lambd, double psi)
{
	this->k_size = k_size;
	
	for(int i=0; i<sigma.size(); i++)
		this->sigma.push_back(sigma.at(i));
	for(int i=0; i<theta.size(); i++)
		this->theta.push_back(theta.at(i));
	
	this->lambd = lambd;
	this->psi = psi;

}

// Function for the Gabor Kernel
Mat CBIR :: gabor_kernel(int ks, double sig, double th, double lm, double ps)
{
	int hks = (ks-1)/2;
    double theta = th*CV_PI/180;
    double psi = ps*CV_PI/180;
    double del = 2.0/(ks-1);
    double lmbd = lm;
    double sigma = sig/ks;
    double x_theta;
    double y_theta;

    Mat kernel(ks,ks, CV_32F);
    
	for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
	
    return kernel;
}

// Color features calculating function
vector<int> CBIR :: color_feature(Mat& src)
{
	Mat hsv_image;
	cvtColor(src,hsv_image,CV_BGR2HSV);
	
	// Image for the quantization
	Mat quant_image = Mat::zeros(hsv_image.rows,hsv_image.cols,CV_8UC1);
	
	// vector for color features with 72x1 bins
	vector<int> col_feat(32,0);
	
	// Color Quantization based on "IMAGE RETRIEVAL USING BOTH COLOR AND TEXTURE FEATURES" FAN-HUI KONG
	for(int i=0; i < hsv_image.rows; ++i)
	{
		for(int j=0; j < hsv_image.cols; ++j)
		{
			
			int h = (int) hsv_image.at<Vec3b>(i,j)[0];
			int s = (int) hsv_image.at<Vec3b>(i,j)[1];
			int v = (int) hsv_image.at<Vec3b>(i,j)[2];
			
			int n;
			if(v<=0.1*255)
			{
				n = 0;
			}
			else if((s<=0.1*255)&&(v>0.1*255)&&(v<=0.4*255))
			{
				n = 1;
			}
			else if((s<=0.1*255)&&(v>0.4*255)&&(v<=0.7*255))
			{
				n = 2;
			}
			else if((s<=0.1*255)&&(v>0.7*255)&&(v<=1*255))
			{
				n = 3;
			}
			else if((h>=0.0/360.0*180 && h<=20.0/360.0*180) || (h>330.0/360.0*180 && h<360.0/360.0*180))
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 4;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 5;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 6;
				}
				else
				{
					n = 7;
				}
			}
			else if(h>20.0/360.0*180&&h<=45.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 8;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 9;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 10;
				}
				else
				{
					n = 11;
				}
			}
			else if(h>45.0/360.0*180&&h<=75.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 12;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 13;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 14;
				}
				else
				{
					n = 15;
				}
			}
			else if(h>75.0/360.0*180&&h<=155.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 16;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 17;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 18;
				}
				else
				{
					n = 19;
				}
			}
			else if(h>155.0/360.0*180&&h<=210.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 20;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 21;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 22;
				}
				else
				{
					n = 23;
				}
			}
			else if(h>210.0/360.0*180&&h<=270.0/360.0*180)
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 24;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 25;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 26;
				}
				else
				{
					n =27;
				}
			}
			else
			{
				if(s<=0.5*255&&v<=0.5*255)
				{
					n = 28;
				}
				else if(s>0.5*255&&v<=0.5*255)
				{
					n = 29;
				}
				else if(s<=0.5*255&&v>0.5*255)
				{
					n = 30;
				}
				else
				{
					n = 31;
				}
			}
			col_feat.at(n) = col_feat.at(n) + 1; 
			quant_image.at<uchar>(i,j) = (uchar)n;
		} 
		
	}
	return col_feat;
}

// Texture feature calculating function
vector<double> CBIR :: texture_feature(Mat& src)
{
	// Converting src image to gray.
	Mat gray, gray_image;
	cvtColor(src,gray,CV_BGR2GRAY);
	gray.convertTo(gray_image,CV_32F, 1.0/255, 0);

	// Vector for the texture feature definition
	vector<double> tex_feat;
	
	for(int i = 0; i < theta.size(); i++)
	{
		for(int j = 0; j < sigma.size(); j++)
		{
			Mat kernel = gabor_kernel(k_size, sigma.at(j), theta.at(i), lambd, psi);
			//cout<<"sigma = "<<sigma.at(j)<<"theta = "<<theta.at(i)<<endl;
			
			// Convolution of gray scale image with gabor kernel
			filter2D(gray_image,gray,CV_32F,kernel);
			
			// Calculating mean and standard deviation 
			Scalar t_mu, t_sigma;
			meanStdDev(gray,t_mu,t_sigma);
			
			//cout<<" Iteration value "<<i*theta.size()+j<<endl;
			tex_feat.push_back(t_mu[0]);
			tex_feat.push_back(t_sigma[0]);
		}
	}
	
	return tex_feat;
}

// Image similarity 
double CBIR :: image_similarity(Mat& query_image, Mat& db_image)
{
	// Image Similarity based on "A Clustering Based Approach to Efficient Image Retrieval" -R. Zhang, and Z. Zhang
	vector<int> q_col_feat,db_col_feat;
	vector<double> q_tex_feat,db_tex_feat;
	
	// get texture feature for query image and DB image
	q_tex_feat = texture_feature(query_image);
	db_tex_feat = texture_feature(db_image);
	
	double d_t = L2dist_texture(db_tex_feat, q_tex_feat); // L2 distance calculation

	// get color feature for query image and DB image
	q_col_feat = color_feature(query_image);
	db_col_feat = color_feature(db_image);
	
	double d_c = HITdist_color(db_col_feat, q_col_feat);  //HIT distance calculation
	
	double img_sim = 0.35*d_t + 0.65*d_c;
	
	return img_sim;
}

// Euclidean distance calculation for texture feature vectors
double CBIR :: L2dist_texture(vector<double>& db_tex_feat, vector<double>& q_tex_feat)
{
	
	double tex_dist = 0;
	
	for(unsigned i=0; i <q_tex_feat.size(); i++)
	{
		tex_dist += pow((db_tex_feat.at(i) - q_tex_feat.at(i)), 2.0);
	}
	
	tex_dist = sqrt(tex_dist);
	
	return tex_dist;
}

// Histogram Intersection Technique for color feature
double CBIR :: HITdist_color(vector<int>& db_col_feat, vector<int>& q_col_feat)
{
	double col_dist = 0;
	
	for(unsigned i=0; i <q_col_feat.size(); i++)
	{
		col_dist += (double) min(q_col_feat.at(i),db_col_feat.at(i));
	}
	
	
	col_dist = col_dist/(double)accumulate(q_col_feat,0); 
	
	return (1 - col_dist);
}


