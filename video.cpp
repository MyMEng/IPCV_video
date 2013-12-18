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
	cv::Mat prev, frame, grey_frame, show_frame;
	
	// Motion tracker
	LKTracker * motionTracker = NULL;

	// Parameter values (if any)
	int threshold = 30;
	std::string videoName; 

	for(int i=1; i < argc; i++)
	{	
		size_t pos;
		std::string argVal(argv[i]);

		if((pos = argVal.find("--threshold=")) != std::string::npos) {

			size_t next_eq = argVal.find_first_of("=", pos);
			size_t next_space = argVal.find_first_of(" \n", next_eq);
			next_eq++;
			std::string thresholdValue = argVal.substr(next_eq, (next_space-1)-next_eq);
			std::cout << "Threshold substr " << thresholdValue << std::endl;
			threshold = atoi(thresholdValue.c_str());
		} else {
			videoName = std::string(argv[i]);
		}
	}
	
	if(!videoName.empty())
	{
		cap.open(videoName);
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
		grey_frame.convertTo(grey_frame, CV_64F);
		prev = grey_frame.clone();
	}
	while(frame.cols == 0);

	double frame_rate = cap.get(CV_CAP_PROP_FPS);
	double frame_msec = 1000 / frame_rate;

	std::cerr << "Frame rate: " << frame_rate << " frame per sec: " << frame_msec << std::endl;


	// Add region to tracker

	int x = frame.cols / 2;
	int y = frame.rows / 2;
	int xl = (int)(0.2*(double)frame.cols);
	int yl = (int)(0.2*(double)frame.rows);


	motionTracker = new LKTracker(threshold);

	motionTracker->AddRegion(cv::Vec2i(x-xl, y-yl), cv::Size(x+xl, y+yl), prev, grey_frame);
	motionTracker->AddRegion(cv::Vec2i(0, 0), cv::Size(150, 150), prev, grey_frame);

	int waitKeyVal = -1;
	
	while(waitKeyVal == -1)
	{
		// Get new frame, remember previous		
		waitKeyVal = cv::waitKey(20);

		prev = grey_frame.clone();
		cap >> frame;
		
		if(!frame.data)
		{
			if(argc > 1)
			{
				if (cap.set(CV_CAP_PROP_POS_FRAMES, 1) == false)
				{
					std::cerr << "Error: unable to rewind" << std::endl;
				}
				continue;
			}
			std::cerr << "Error: no frame data." << std::endl;
			break;
		}

		
		// Convert frame to grey-scale
		cv::cvtColor(frame, grey_frame, CV_BGR2GRAY);
		grey_frame.convertTo(grey_frame, CV_64F);

		// Update region
		motionTracker->Update(prev, grey_frame);
		
		// Show current frame
		//motionTracker.ShowAll();

		show_frame = frame.clone();

		show_frame.convertTo(show_frame, CV_8U);

		motionTracker->ShowMotion(show_frame);
		
		imshow("Video", show_frame);
	}

	if(motionTracker != NULL)
	{
		delete motionTracker;
		motionTracker = NULL;
	}
}
