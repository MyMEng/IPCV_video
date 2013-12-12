#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <iostream>

#include "motion.hpp"

// LKTracker constructor
LKTracker::LKTracker()
{
}

LKTracker::~LKTracker()
{
	// Update all tracked region
	for(MotionVector::iterator iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motion = *iter;

		if(motion != NULL)
		{
			delete motion;
			motion = NULL;
		}
	}
	this->regions.clear();
}

// Add region to tracking
void LKTracker::AddRegion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next)
{
	Motion * motionRegion = new Motion(position, regionSize, frame, next);
	this->regions.push_back(motionRegion);

	int num_regions = regions.size();

	std::stringstream xStream, yStream, tStream;
	xStream << num_regions << " X derivative";
	yStream << num_regions << " Y derivative";
	tStream << num_regions << " T derivative";

	std::string x, y, t;
	x = xStream.str();
	y = yStream.str();
	t = tStream.str();

	cv::namedWindow(x, 2);
	cv::namedWindow(y, 2);
	cv::namedWindow(t, 2);
	cv::namedWindow("Vx", 3);
	cv::namedWindow("Vy", 4);

	motionRegion->SetWindowNames(x, y, t);
}

// Update with new frames
void LKTracker::Update(cv::Mat& frame, cv::Mat& next)
{
	MotionVector::iterator iter;

	// Update all tracked region
	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motionRegion = *iter;
		motionRegion->Update(frame, next);
	}
}

void LKTracker::ShowAll()
{
	MotionVector::iterator iter;

	// For each region, show derivatives
	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motion = (*iter);
		cv::imshow(motion->getWindowTitleX(), motion->getIx());
		cv::imshow(motion->getWindowTitleY(), motion->getIy());
		cv::imshow(motion->getWindowTitleT(), motion->getIt());
	}
}

void LKTracker::ShowMotion()
{
	MotionVector::iterator iter;

	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motion = (*iter);
		cv::imshow("Vx", motion->getVx());
		cv::imshow("Vy", motion->getVy());
	}
}

Motion::Motion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next)
{
	int x = position[0];
	int y = position[1];

	// Check if this region is possible
	assert(x >= 0 && y >= 0 && x < frame.cols && y < frame.rows);

	// Make sure regions are not too big
	if(x + regionSize.width >= frame.cols)
	{
		regionSize.width = frame.cols - x;
	}

	if(y+regionSize.height >= frame.rows)
	{	
		regionSize.height = frame.rows - y;
	}
	
	// Create rectangle to extract region later
	this->region = cv::Rect(x, y, regionSize.width, regionSize.height);

	// Initialize derivative
	this->derivative = new Derivative(this->region.height, this->region.width);	
}

Motion::~Motion()
{
	if(this->derivative != NULL)
	{
		delete this->derivative;
		this->derivative = NULL;
	}
}

void Motion::Update(cv::Mat& frame, cv::Mat& next)
{
	this->extractRegionAndUpdate(frame, next);
	this->derivative->computeVelocity();
}

void Motion::extractRegionAndUpdate(cv::Mat& frame, cv::Mat& next)
{
	this->extractedFrame = frame(this->region);
	this->extractedNext = next(this->region);

	this->derivative->setDerivatives(extractedFrame, extractedNext);
}

void Motion::SetWindowNames(std::string& xWindowTitle, std::string& yWindowTitle, std::string& tWindowTitle)
{
	this->xWindowTitle = xWindowTitle;
	this->yWindowTitle = yWindowTitle;
	this->tWindowTitle = tWindowTitle;
}

std::string& Motion::getWindowTitleX()
{
	return this->xWindowTitle;
}
std::string& Motion::getWindowTitleY()
{
	return this->yWindowTitle;
}

std::string& Motion::getWindowTitleT()
{
	return this->tWindowTitle;
}