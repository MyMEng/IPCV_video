#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>

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
	this->regions.push_back(new Motion(position, regionSize, frame, next));

	int num_regions = regions.size();

	std::ostringstream xStream, yStream, tStream;
	xStream << num_regions << " X derivative";
	cv::namedWindow(xStream.str(), num_regions*10);

	yStream << num_regions << " Y derivative";
	cv::namedWindow(yStream.str(), num_regions*10+1);

	tStream << num_regions << " T derivative";
	cv::namedWindow(tStream.str(), num_regions*10+2);
	
	this->regions.at(num_regions-1)->SetWindowNames(xStream.str(), yStream.str(), tStream.str());
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

Motion::Motion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next)
{
	int x = position[0];
	int y = position[1];

	// Check if this region is possible
	assert(x >= 0 && y >= 0 && x < frame.cols && y < frame.rows);
	assert(x + regionSize.width < frame.cols && y+regionSize.height < frame.rows);

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
