#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "derivative.hpp"

Derivative::Derivative(int rows, int cols) : ddepth(CV_64F)
{
	this->ix = cv::Mat(rows, cols, ddepth);
	this->iy = cv::Mat(rows, cols, ddepth);
	this->it = cv::Mat(rows, cols, ddepth);	
	
	this->vx = cv::Mat(rows, cols, CV_32S);	
	this->vy = cv::Mat(rows, cols, CV_32S);	
}

Derivative::~Derivative()
{
	this->ix.release();
	this->iy.release();
	this->it.release();
}

// Compute derivatives
void Derivative::setDerivatives(cv::Mat& current_frame, cv::Mat& next_frame)
{
	this->computeX(current_frame, next_frame);
	this->computeY(current_frame, next_frame);
	this->computeT(current_frame, next_frame);
}

cv::Mat& Derivative::getIx()
{
	return this->ix;
}
	
cv::Mat& Derivative::getIy()
{
	return this->iy;
}

cv::Mat& Derivative::getIt()
{
	return this->it;
}

cv::Mat& Derivative::getVx()
{
	return this->vx;
}

cv::Mat& Derivative::getVy()
{
	return this->vy;
}

void Derivative::computeX(cv::Mat& frame, cv::Mat& next)
{
 	for(int i = 0; i < frame.rows; ++i)
 	{
 		for (int j = 1; j < frame.cols-1; ++j)
 		{
 			double first = frame.at<double>(i, j-1);
 			double second = frame.at<double>(i, j+1);
 			double diff = second - first;
			this->ix.at<double>(i, j) = diff;
 		}
 	}
 }

void Derivative::computeY(cv::Mat& frame, cv::Mat& next)
{
 	for(int i = 1; i < frame.rows-1; ++i)
 	{
 		for (int j = 0; j < frame.cols; ++j)
 		{
 			double first = frame.at<double>(i-1, j);
 			double second = frame.at<double>(i+1, j);
 			double diff = second - first;
			this->iy.at<double>(i, j) = diff;
 		}
 	}
}

void Derivative::computeT(cv::Mat& frame, cv::Mat& next)
{
	for(int i = 0; i < frame.rows; ++i)
 	{
 		for (int j = 0; j < frame.cols; ++j)
 		{
 			double first = frame.at<double>(i, j);
 			double second = next.at<double>(i, j);
 			double diff = second - first;
			this->it.at<double>(i, j) = diff;
 		}
 	}
}

void Derivative::computeVelocity()
{
	cv::Mat A, b, V, Vconverted;

	for(int i = 0; i < this->ix.rows; i += 10)
	{
		for (int j = 0; j < this->ix.cols; j += 10)
		{
			A = cv::Mat::zeros(2, 2, CV_64FC1);
			b = cv::Mat::zeros(2, 1, CV_64FC1);
			V = cv::Mat::zeros(2, 1, CV_64FC1);

			double x = ix.at<double>(i, j);
			double y = iy.at<double>(i, j);
			double t = it.at<double>(i, j);

			A.at<double>(0,0) = x * x;
			A.at<double>(0,1) = x * y; 
			A.at<double>(1,0) = x * y;
			A.at<double>(1,1) = y * y;

			b.at<double>(0,0) = -t * x;
			b.at<double>(1,0) = -t * y;

			V = A.inv() * b;
			V.convertTo(Vconverted, CV_32S);

			this->vx.at<int>(i, j) = Vconverted.at<int>(0,0);
			this->vy.at<int>(i, j) = Vconverted.at<int>(1,0);
		}
	}
}