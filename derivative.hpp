#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Derivative 
{
private:
	cv::Mat ix, iy, it, vx, vy, xd, yd;

	const int ddepth;

	void computeX(cv::Mat& frame, cv::Mat& next);
	void computeY(cv::Mat& frame, cv::Mat& next);
	void computeT(cv::Mat& frame, cv::Mat& next);
	void applyDerivative(cv::Mat& in, cv::Mat& out, cv::Mat& kernel);
	void averageTwoFrames(cv::Mat& out, cv::Mat& first, cv::Mat& second);
public:

	Derivative(int rows, int cols);
	~Derivative();

	void setDerivatives(cv::Mat& current_frame, cv::Mat& next_frame);
	void computeVelocity();

	cv::Mat& getIx();
	cv::Mat& getIy();
	cv::Mat& getIt();
	cv::Mat& getVx();
	cv::Mat& getVy();
};

#endif