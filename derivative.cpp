#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "derivative.hpp"

Derivative::Derivative(int rows, int cols, int regionSize) : ddepth(CV_64F), regionSize(regionSize)
{
	this->ix = cv::Mat::zeros(rows, cols, ddepth);
	this->iy = cv::Mat::zeros(rows, cols, ddepth);
	this->it = cv::Mat::zeros(rows, cols, ddepth);	
	
	this->vx = cv::Mat::zeros(rows, cols, ddepth);	
	this->vy = cv::Mat::zeros(rows, cols, ddepth);


	const int kernelSize = 7;
	const int xder[kernelSize][3] = {
		{-1, 0, 1},
		{-1, 0, 1},
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
		{-1, 0, 1},
		{-1, 0, 1}
	};
	const int yder[3][kernelSize] = {
		{-1, -1, -1, -2, -1, -1, -1},
		{0,  0 , 0 , 0 , 0 , 0 , 0},
		{1,  1 , 1 , 2 , 1 , 1 , 1}
	};

	this->xd = cv::Mat::zeros(kernelSize, kernelSize, ddepth);	
	this->yd = cv::Mat::zeros(kernelSize, kernelSize, ddepth);

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			this->yd.at<double>(i,j) = yder[i][j];
		}
	}
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			this->xd.at<double>(i,j) = xder[i][j];
		}
	}
}

Derivative::~Derivative()
{
	this->ix.release();
	this->iy.release();
	this->it.release();
	this->xd.release();
	this->yd.release();
	this->vx.release();
	this->vy.release();
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

void Derivative::applyDerivative(cv::Mat& in, cv::Mat& out, cv::Mat& kernel)
{
	// initialize the output using the input size
	out.create(in.size(), CV_64F) ;

	// create a padded version of the input to avoid border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( in, paddedInput, kernelRadiusX, kernelRadiusX,
		kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE );

	// convolve
	for ( int i = 0; i < in.rows; i++ )
	{	
		for( int j = 0; j < in.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indexes we are using
					int imagex = i + kernelRadiusX + m;
					int imagey = j + kernelRadiusY + n;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					double imageval = paddedInput.at<double>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			out.at<double>(i, j) = (double)sum;
		}
	}
}

void Derivative::averageTwoFrames(cv::Mat& out, cv::Mat& first, cv::Mat& second)
{
	for ( int i = 0; i < first.rows; i++ )
	{	
		for( int j = 0; j < first.cols; j++ )
		{
			double f = first.at<double>(i, j);
			double s = second.at<double>(i, j);
			out.at<double>(i, j) = (f+s)/2;
		}
	}
}

void Derivative::computeX(cv::Mat& frame, cv::Mat& next)
{
	cv::Mat thisFrame, nextFrame;
	this->applyDerivative(frame, thisFrame, this->xd);
	this->applyDerivative(next, nextFrame, this->xd);
	this->averageTwoFrames(this->ix, thisFrame, nextFrame);
}

void Derivative::computeY(cv::Mat& frame, cv::Mat& next)
{
	cv::Mat thisFrame, nextFrame;
 	this->applyDerivative(frame, thisFrame, this->yd);
 	this->applyDerivative(next, nextFrame, this->yd);
 	this->averageTwoFrames(this->iy, thisFrame, nextFrame);
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

	const double magnitudeScale = 30.0;

	for(int i = 0; i < this->ix.rows - regionSize; i += regionSize)
	{
		for (int j = 0; j < this->ix.cols - regionSize; j += regionSize)
		{

			A = cv::Mat::zeros(2, 2, CV_64FC1);
			b = cv::Mat::zeros(2, 1, CV_64FC1);
			V = cv::Mat::zeros(2, 1, CV_64FC1);

			// Sum over region
			for(int k = 0; k < regionSize; ++k) 
			{
				for(int l = 0; l < regionSize; ++l) 
				{
					double x = ix.at<double>(i+k, j+l);
					double y = iy.at<double>(i+k, j+l);
					double t = it.at<double>(i+k, j+l);

					A.at<double>(0,0) += x * x;
					A.at<double>(0,1) += x * y; 
					A.at<double>(1,0) += x * y;
					A.at<double>(1,1) += y * y;

					b.at<double>(0,0) += -t * x;
					b.at<double>(1,0) += -t * y;
				}
			}
			//std::cout << " A " << A << " b " << b << std::endl;
			V = A.inv() * b;

			//std::cout << " A.inv() " << A.inv() << " V " << V << std::endl;

			this->vx.at<double>(i+regionSize/2, j+regionSize/2) = magnitudeScale * V.at<double>(0,0);
			this->vy.at<double>(i+regionSize/2, j+regionSize/2) = magnitudeScale * V.at<double>(1,0);
		}
	}
}