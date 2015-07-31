/************************************/
/*       Testing Program            */
/************************************/

#include <iostream>
#include "CBIR.h"
#include <vector>
#include <boost/assign/std/vector.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using std::vector;
using namespace boost::assign;
int main()
{
	int k_size = 21;
	vector<double> sigma ;
	sigma += 0.05, 0.10, 0.15, 0.20;
	vector<double> theta;
	theta += 0, CV_PI/6, CV_PI/3, CV_PI/2, 2*CV_PI/3, 5*CV_PI/6; 
	double lambd = 50; 
	double psi = CV_PI/2;

	// Calling constructor
	CBIR cbir(k_size, sigma, theta, lambd, psi);
	
	Mat db_image = imread("../../CBIR/src/kajal1.jpg",1);
	if(!db_image.data ) // Check for invalid input
    {
	  cout <<"Could not open or find the image" <<endl ;
	  return -1;
	}
	
	Mat q_image = imread("../../CBIR/src/kajal2.jpg",1);
	if(!q_image.data ) // Check for invalid input
    {
	  cout <<"Could not open or find the image" <<endl ;
	  return -1;
	}
	
	double img_sim = cbir.image_similarity(q_image, db_image);
	
	cout<<"Image Similarity = "<<img_sim<<endl;
	
	return 0;
}
