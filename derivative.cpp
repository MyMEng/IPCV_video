#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "derivative.hpp"


void Derivative::setDerivatives(cv::Mat frame)
{

}

cv::Mat Derivative::getIx()
{
	return this->ix;
}
	
cv::Mat Derivative::getIy()
{
	return this->iy;
}

cv::Mat Derivative::getIt()
{
	return this->it;
}