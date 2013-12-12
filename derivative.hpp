#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Derivative 
{
private:
	cv::Mat ix, iy, it, vx, vy;

	const int ddepth;

	void computeX(cv::Mat& frame, cv::Mat& next);
	void computeY(cv::Mat& frame, cv::Mat& next);
	void computeT(cv::Mat& frame, cv::Mat& next);

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