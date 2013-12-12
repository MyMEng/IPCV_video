#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdio>

#include "derivative.hpp"
#include "motion.hpp"

int main( int argc, const char** argv )
{
	// Capture device
	cv::VideoCapture cap;

	// Frames
	cv::Mat prev, frame, grey_frame;
	
	// Motion tracker
	LKTracker motionTracker;
	
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
	
	// Create windows
	cv::namedWindow("Video", 1);

	// Get first two farmes
	do
	{	
		cap >> frame;

		cv::cvtColor(frame, grey_frame, CV_BGR2GRAY);
		grey_frame.convertTo(grey_frame, CV_8U);
		prev = grey_frame.clone();
	}
	while(frame.cols == 0);

	double frame_rate = cap.get(CV_CAP_PROP_FPS);
	double frame_msec = 1000 / frame_rate;

	std::cerr << "Frame rate: " << frame_rate << " frame per sec: " << frame_msec << std::endl;


	// Add region to tracker
	motionTracker.AddRegion(cv::Vec2i(0, 0), cv::Size(120, 120), prev, grey_frame);
	//motionTracker.AddRegion(cv::Vec2i(150, 50), cv::Size(100, 100), prev, grey_frame);

	for(;;)
	{
		// Get new frame, remember previous		
		cv::waitKey(20);
		prev = grey_frame.clone();
		cap >> frame;
		
		if(!frame.data)
		{
			if(argc > 1)
			{
				if (cap.set(CV_CAP_PROP_POS_FRAMES, 0) == false)
				{
					std::cerr << "Error: unable to rewind" << std::endl;
				}
				continue;
			}
			std::cerr << "Error: no frame data." << std::endl;
			break;
		}

		imshow("Video", frame);
		
		// Convert frame to grey-scale
		cv::cvtColor(frame, grey_frame, CV_BGR2GRAY);

		// Update region
		motionTracker.Update(prev, grey_frame);
		
		// Show current frame
		motionTracker.ShowAll();
		motionTracker.ShowMotion();
	}
}
