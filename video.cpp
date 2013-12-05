#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdio>

int main( int argc, const char** argv )
{
	cv::VideoCapture cap;
	cv::Mat frame, edges;
	
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
	
	for(;;)
	{
		cv::waitKey(20);
		
		cap >> frame;
		
		if(!frame.data)
		{
			std::cerr << "Error: no frame data." << std::endl;
			break;
		}

		// Convert frame to grey-scale
		cv::cvtColor(frame, edges, CV_BGR2GRAY);

		// Blur the frame
		cv::GaussianBlur(edges, edges, cv::Size(3,3), 1.5, 1.5);

		// Show current frame
		cv::imshow("Video", edges);
	}
}
