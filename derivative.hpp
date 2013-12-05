#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Derivative 
{
private:
	cv::Mat ix, iy, it;

public:

	void setDerivatives(cv::Mat frame);

	cv::Mat getIx();
	cv::Mat getIy();
	cv::Mat getIt();
};

#endif