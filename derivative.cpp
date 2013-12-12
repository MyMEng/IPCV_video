#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "derivative.hpp"

Derivative::Derivative(int rows, int cols) : ddepth(CV_8U)
{
	this->ix = cv::Mat(rows, cols, ddepth);
	this->iy = cv::Mat(rows, cols, ddepth);
	this->it = cv::Mat(rows, cols, ddepth);	
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

void Derivative::computeX(cv::Mat& frame, cv::Mat& next)
{
 	for(int i = 0; i < frame.rows; ++i)
 	{
 		for (int j = 1; j < frame.cols-1; ++j)
 		{
 			int first = static_cast<int>(frame.at<unsigned char>(i, j-1));
 			int second = static_cast<int>(frame.at<unsigned char>(i, j+1));
 			int diff = std::abs(second - first);

 			// Truncate
 			if(diff > 255) {
 				diff = 255;
 			}

			this->ix.at<unsigned char>(i, j) = static_cast<unsigned char>(diff);
 		}
 	}
 }

void Derivative::computeY(cv::Mat& frame, cv::Mat& next)
{
 	for(int i = 1; i < frame.rows-1; ++i)
 	{
 		for (int j = 0; j < frame.cols; ++j)
 		{
 			int first = static_cast<int>(frame.at<unsigned char>(i-1, j));
 			int second = static_cast<int>(frame.at<unsigned char>(i+1, j));
 			int diff = std::abs(second - first);

 			// Truncate
 			if(diff > 255) {
 				diff = 255;
 			}

			this->iy.at<unsigned char>(i, j) = static_cast<unsigned char>(diff);
 		}
 	}
}

void Derivative::computeT(cv::Mat& frame, cv::Mat& next)
{
	for(int i = 0; i < frame.rows; ++i)
 	{
 		for (int j = 0; j < frame.cols; ++j)
 		{
 			int first = static_cast<int>(frame.at<unsigned char>(i, j));
 			int second = static_cast<int>(next.at<unsigned char>(i, j));
 			int diff = std::abs(second - first);

 			// Truncate
 			if(diff > 255) {
 				diff = 255;
 			}

			this->it.at<unsigned char>(i, j) = static_cast<unsigned char>(diff);
 		}
 	}
}