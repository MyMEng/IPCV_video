#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "derivative.hpp"


Derivative::Derivative() : ddepth(CV_16S)
{
}

Derivative::~Derivative()
{
}

// Compute derivatives
void Derivative::setDerivatives(cv::Mat current_frame, cv::Mat next_frame)
{
	this->computeX(current_frame, next_frame);
	this->computeY(current_frame, next_frame);
	this->computeT(current_frame, next_frame);
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

void Derivative::computeX(cv::Mat frame, cv::Mat next)
{
	cv::Mat x_grad, x_grad_next, x_abs, x_abs_next;

	cv::Sobel(frame, x_grad, this->ddepth, 1, 0); 
	cv::Sobel(next, x_grad_next, this->ddepth, 1, 0);

	cv::convertScaleAbs( x_grad, x_abs);
	cv::convertScaleAbs( x_grad_next, x_abs_next );

	cv::addWeighted(x_abs, 0.5, x_abs_next, 0.5, 0.0, this->ix);
}

void Derivative::computeY(cv::Mat frame, cv::Mat next)
{
	cv::Mat grad, grad_next, abs_frame, abs_next;

	cv::Sobel(frame, grad, this->ddepth, 0, 1); 
	cv::Sobel(next, grad_next, this->ddepth, 0, 1);

	cv::convertScaleAbs( grad, abs_frame);
	cv::convertScaleAbs( grad_next, abs_next );

	cv::addWeighted(abs_frame, 0.5, abs_next, 0.5, 0.0, this->iy);
}

void Derivative::computeT(cv::Mat frame, cv::Mat next)
{
	cv::absdiff(frame, next, this->it);
}