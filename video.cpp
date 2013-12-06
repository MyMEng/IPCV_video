#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdio>

#include "derivative.hpp"
#include "motion.hpp"

int main( int argc, const char** argv )
{
	cv::VideoCapture cap;
	cv::Mat prev, frame, grey_frame;
	Derivative der;
	
	if(argc > 1)
	{
		cap.open(std::string(argv[1]));
	}
	else
	{
		cap.open(CV_CAP_ANY);
	}
	
	if(!cap.isOpened())
	{
		std::cerr << "Error: could not load a camera or video." << std::endl;
	}
	
	// Create window
	cv::namedWindow("Video", 1);
	cv::namedWindow("Xderivative", 2);
	cv::namedWindow("Yderivative", 3);
	cv::namedWindow("Tderivative", 4);
	
	cv::waitKey(20);
	cap >> frame;
	cv::cvtColor(frame, grey_frame, CV_BGR2GRAY);

	for(;;)
	{
		
		cv::waitKey(20);
		prev = grey_frame.clone();
		
		cap >> frame;
		
		if(!frame.data)
		{
			std::cerr << "Error: no frame data." << std::endl;
			break;
		}

		// Convert frame to grey-scale
		cv::cvtColor(frame, grey_frame, CV_BGR2GRAY);

		der.setDerivatives(prev, grey_frame);

		// Show current frame
		cv::imshow("Video", grey_frame);
		cv::imshow("Xderivative", der.getIx());
		cv::imshow("Yderivative", der.getIy());
		cv::imshow("Tderivative", der.getIt());
	}
}
