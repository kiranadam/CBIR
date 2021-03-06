/***********************************************************/
/*      Content Based Image Retrival System  Header        */
/*            Author : Kirankumar V. Adam                  */
/***********************************************************/

#ifndef CBIR_H
#define CBIR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

class CBIR
{
	private :
	    // variables for the Gabor function for 6 orientations and 4 scales
		int k_size; // kernel size 
		vector<double> sigma; //Standard deviation for 4 different scales
		vector<double> theta; // Orientations 
		double lambd; // Wavelength of sinusoidal factor
		double psi; // Phase offset.
		//Mat src; // Image source		
		
		// function definition for gabor kernel
		Mat gabor_kernel(int ks, double sig, double th, double lm, double ps); 
		
		// Euclidean distance measurement for texture feature
		double L2dist_texture(vector<double>& db_tex_feat, vector<double>& q_tex_feat);
		
		// Histogram Intersection Technique for color feature
		double HITdist_color(vector<int>& db_col_feat, vector<int>& q_col_feat);
		
		// color feature 
		vector <int> color_feature(Mat& src);
		// texture feature
		vector <double> texture_feature(Mat& src);
		
	
	public :
		// Constructor for CBIR
		CBIR(int k_size, vector<double>& sigma, vector<double>& theta, double lambd, double psi);
		
		// Image similarity calculation 
		double image_similarity(Mat& query_image, Mat& db_image);
};

#endif 
