#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>

#include "motion.hpp"

// Add region to tracking
void LKTracker::AddRegion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next)
{
	this->regions.push_back(Motion(position, regionSize, frame, next));
}

// Update with new frames
void LKTracker::Update(cv::Mat& frame, cv::Mat& next)
{
	MotionVector::iterator iter;

	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		(*iter).Update(frame, next);
	}
}

void LKTracker::ShowAll()
{
	MotionVector::iterator iter;

	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion motion = (*iter);
		cv::imshow("Xderivative", motion.getIx());
		cv::imshow("Yderivative", motion.getIy());
		cv::imshow("Tderivative", motion.getIt());
	}
}

Motion::Motion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next)
{
	int x = position[0];
	int y = position[1];

	assert(x >= 0 && y >= 0 && x < frame.cols && y < frame.rows);
	assert(x + regionSize.width < frame.cols && y+regionSize.height < frame.rows);

	this->region = cv::Rect(x, y, regionSize.width, regionSize.height);

	this->derivative = new Derivative();	
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
}

void Motion::extractRegionAndUpdate(cv::Mat& frame, cv::Mat& next)
{
	this->extractedFrame = frame(this->region);
	this->extractedNext = next(this->region);

	this->derivative->setDerivatives(extractedFrame, extractedNext);
}

