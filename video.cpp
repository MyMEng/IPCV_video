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

	// Threshold
	int threshold = -1;

	// Video name
	std::string videoName;

	// Allow defining regions?
	bool askForRegions = false;

	// Display all unit vectors for motion?
	bool showAllVectors = false;

	int windowSize = 30;

	// Show derivatives
	bool showDerivs = false;

	// Scan parameters read
	for(int i=1; i < argc; i++)
	{	
		size_t pos, next_eq, next_space;
		std::string argVal(argv[i]);

		if((pos = argVal.find("--threshold=")) != std::string::npos) {

			next_eq = argVal.find_first_of("=", pos);
			next_space = argVal.find_first_of(" \n", next_eq);
			next_eq++;
			std::string thresholdValue = argVal.substr(next_eq, (next_space-1)-next_eq);
			threshold = atoi(thresholdValue.c_str());
		} else if((pos = argVal.find("--regions")) != std::string::npos) {
			askForRegions = true;
		} else if((pos = argVal.find("--showall")) != std::string::npos) {
			showAllVectors = true;
			askForRegions = false;
		} else if((pos = argVal.find("--showderiv")) != std::string::npos) {
			showDerivs = true;
		}else if((pos = argVal.find("--regionSize=")) != std::string::npos) {

			size_t next_eq = argVal.find_first_of("=", pos);
			size_t next_space = argVal.find_first_of(" \n", next_eq);
			
			next_eq++;
			std::string regionValue = argVal.substr(next_eq, (next_space-1)-next_eq);
			windowSize = atoi(regionValue.c_str());
		}
		else {
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

	std::cout << "Frame size. Cols " << frame.cols << " Rows: " << frame.rows << std::endl;


	// Add region to tracker

	int x = frame.cols / 2;
	int y = frame.rows / 2;
	int xl = (int)(0.2*(double)frame.cols);
	int yl = (int)(0.2*(double)frame.rows);

	// Set default theshold, sqrt(area) * 0.1
	if(threshold == -1) 
	{
		int area = frame.cols * frame.rows;
		double value = 1.7237*exp(0.00001012 * area); 
		threshold = static_cast<int>(value);
	}

	// Just make sure if it not zero
	if(threshold < 1) {
		threshold = 5;
	}

	motionTracker = new LKTracker(threshold, showDerivs);

	if(!askForRegions) 
	{	
		if(showAllVectors) 
		{
			motionTracker->AddRegion(cv::Vec2i(0, 0), cv::Size(frame.cols, frame.rows), prev, grey_frame, windowSize);
		} 
		else
		{
			motionTracker->AddRegion(cv::Vec2i(x-xl, y-yl), cv::Size(x+xl, y+yl), prev, grey_frame, windowSize);
			motionTracker->AddRegion(cv::Vec2i(0, 0), cv::Size(150, 150), prev, grey_frame, windowSize);
		}
	} else 
	{

		int region_no = 1;

		std::cout << "How many regions to define?" << std::endl;
		std::cin >> region_no;

		std::cout << "Width and height should be at least 64, otherwise it will be stretched" << std::endl;

		for(int i = 0; i < region_no; i++) 
		{
			int x, y, w, h;
			std::cout << "Enter regions x y width height: ";
			std::cin >> x >> y >> w >> h;

			if(w < 64) w = 64;
			if(h < 64) h = 64;

			motionTracker->AddRegion(cv::Vec2i(x,y), cv::Size(w, h), prev, grey_frame, windowSize); 
		}
	}

	int waitKeyVal = -1;

	while(waitKeyVal == -1)
	{
		// Get new frame, remember previous		
		waitKeyVal = cv::waitKey(16);

		prev = grey_frame.clone();
		cap >> frame;

		cv::medianBlur(frame, frame, 3);
		
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
		if(showDerivs) {
			motionTracker->ShowAll();
		}

		show_frame = frame.clone();

		show_frame.convertTo(show_frame, CV_8U);

		if(showAllVectors) {
			motionTracker->ShowAllVectors(show_frame);	
		} else {
			motionTracker->ShowMotion(show_frame);
		}
		
		imshow("Video", show_frame);
	}

	if(motionTracker != NULL)
	{
		delete motionTracker;
		motionTracker = NULL;
	}
}
