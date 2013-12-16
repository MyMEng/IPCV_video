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

		cv::Mat a, b, c;
		motion->getIx().convertTo(a, CV_8U);
		motion->getIy().convertTo(b, CV_8U);
		motion->getIt().convertTo(c, CV_8U);
		cv::normalize(a, a, 0, 255, cv::NORM_MINMAX);
		cv::normalize(b, b, 0, 255, cv::NORM_MINMAX);
		cv::normalize(c, c, 0, 255, cv::NORM_MINMAX);

		cv::imshow(motion->getWindowTitleX(), a);
		cv::imshow(motion->getWindowTitleY(), b);
		cv::imshow(motion->getWindowTitleT(), c);
	}
}

void LKTracker::ShowMotion(cv::Mat& image)
{
	MotionVector::iterator iter;

	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motion = (*iter);
		for(int i = 0; i < motion->getVx().rows; i++)
		{
			for(int j = 0; j < motion->getVy().rows; j++)
			{
				double x_component = motion->getVx().at<double>(i,j);
				double y_component = motion->getVy().at<double>(i,j);

				//std::cout << "vx " << x_component << " vy " << y_component << std::endl;

				double magnitude_treshold = 50;

				cv::Point p1 = cv::Point(j, i);
				cv::Point p2 = cv::Point(j+x_component, i+y_component);

				// cv::Point px1 = cv::Point(j, i);
				// cv::Point px2 = cv::Point(j+x_component, i);

				// cv::Point py1 = cv::Point(j, i);
				// cv::Point py2 = cv::Point(j, i+y_component);

				// distance ?????
				// if(cv::norm(px1-px2) < magnitude_treshold || cv::norm(py1-py2) < magnitude_treshold)
				if(cv::norm(p1-p2) < magnitude_treshold)
					continue;

				cv::circle ( image , p1 , 20 , cv::Scalar(0,255,0) , 2 , 8 );
				cv::line(image, p1, p2, CV_RGB(255, 0, 0), 2);
				cv::circle ( image , p2 , 5 , cv::Scalar(0,255,0) , 2 , 8 );

				// check whether the motion is big enough
				// if (cv::norm(px1-px2) > 100)
				// {
							// cv::line(image, px1, px2, CV_RGB(255, 0, 0), 2);
				// }
 			// 	if (cv::norm(py1-py2) > 100)
				// {
							// cv::line(image, py1, py2, CV_RGB(255, 0, 0), 2);
				// }

				// std::cout << "P1 " << p1 << " P2 " << p2 << std::endl;
				// Motion::detectMotion(p1, p2);
					// disregard motion in Y
	if (p1.y-p2.y < 50)
	{

		if (p1.x-p2.x > 50)
		{
			std::cout << "LEFT" << std::endl;
		}

		if (p1.x-p2.x < -50)
		{
			std::cout << "RIGHT" << std::endl;
		}
	}

			}
		}
	}
}

void Motion::detectMotion(cv::Point A, cv::Point B)
{
	// disregard motion in Y
	if (A.y-B.y < 50)
	{

		if (A.x-B.x > 20)
		{
			std::cout << "LEFT" << std::endl;
		}

		if (A.x-B.x < -20)
		{
			std::cout << "RIGHT" << std::endl;
		}
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