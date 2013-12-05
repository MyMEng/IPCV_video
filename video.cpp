#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdio>

int main( int argc, const char** argv )
{
	cv::VideoCapture cap;
	
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
	
	cv::Mat frame, edges;
	cv::namedWindow("video", 1);
	
	for(;;)
	{
		cv::waitKey(20);
		
		cap >> frame;

		cv::cvtColor(frame, edges, CV_BGR2GRAY);
		cv::GaussianBlur(edges, edges, cv::Size(7,7), 1.5, 1.5);
		
		if(!frame.data)
		{
			std::cerr << "Error: no frame data." << std::endl;
			break;
		}

		cv::imshow("video", edges);
	}
}
