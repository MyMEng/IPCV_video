#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <iostream>

#include "motion.hpp"

// LKTracker constructor
LKTracker::LKTracker(int threshold, bool showDerivs) 
	: magnitude_treshold(threshold), showDerivs(showDerivs)
{
	std::cout << "Using threshold " << threshold << std::endl;
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
void LKTracker::AddRegion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next, int windowSize)
{
	Motion * motionRegion = new Motion(position, regionSize, frame, next, windowSize);
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

	if(this->showDerivs) 
	{
		cv::namedWindow(x, 2);
		cv::namedWindow(y, 2);
		cv::namedWindow(t, 2);
	}

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

void LKTracker::ShowAllVectors(cv::Mat& image) 
{
		MotionVector::iterator iter;

	int motionR, motionT;
	double motionSumR, motionSumT;

	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motion = (*iter);

		motionR = 0;
		motionT = 0;
		motionSumR = 0;
		motionSumT = 0;

		cv::Rect origin = motion->getRect();

		cv::rectangle(image, origin, CV_RGB(255, 0, 255), 2);

		for(int i = 0; i < motion->getVx().rows; i++)
		{
			for(int j = 0; j < motion->getVx().cols; j++)
			{
				double x_component = motion->getVx().at<double>(i,j);
				double y_component = motion->getVy().at<double>(i,j);

				if(x_component == 0.0 && y_component == 0.0)
					continue;

				cv::Point p1 = cv::Point(j+origin.x, i+origin.y);
				cv::Point p2 = cv::Point(j+origin.x, i+origin.y);

				if(std::abs(x_component) >= 10 && std::abs(x_component) > std::abs(y_component))
				{
					if(std::abs(x_component) > motion->getWindowSize()) 
					{
						x_component = (x_component > 0) ? motion->getWindowSize()/2 : -motion->getWindowSize()/2;
					}
					p2.x += x_component;

					if(std::abs(y_component) > motion->getWindowSize()) 
					{
						y_component = (y_component > 0) ? motion->getWindowSize()/2 : -motion->getWindowSize()/2;
					}
					p2.y += y_component;


					motionSumR += x_component;
					motionR++;
				}

				if(std::abs(y_component) >= 10 && std::abs(y_component) > std::abs(x_component))
				{
					if(std::abs(x_component) > motion->getWindowSize()) 
					{
						x_component = (x_component > 0) ? motion->getWindowSize()/2 : -motion->getWindowSize()/2;
					}
					p2.x += x_component;

					if(std::abs(y_component) > motion->getWindowSize()) 
					{
						y_component = (y_component > 0) ? motion->getWindowSize()/2 : -motion->getWindowSize()/2;
					}
					p2.y += y_component;


					motionSumT += y_component;
					motionT++;
				}

				cv::line(image, p1, p2, CV_RGB(0, 0, 255), 1);
			}
		}

		double av_motion_r = motionSumR / motionR,
		av_motion_l = motionSumT / motionT;

		cv::Point p(origin.x+50, origin.y+50);
		cv::Point p1(origin.x+50, origin.y+75);

		std::string text;
		std::ostringstream s;

		if(av_motion_r == av_motion_r) 
		{
			if (av_motion_r > 0) 
			{
				cv::putText(image, "L", p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
			}
			else if (av_motion_r < 0)
			{
				cv::putText(image, "R", p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
			}

			s << "(" << av_motion_r << ")";
			cv::putText(image, s.str(), p1, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 2);
		}

		p.x+=100;
		p1.x+=100;


		if(av_motion_l == av_motion_l) 
		{
			if (av_motion_l > 0) 
			{
				cv::putText(image, "D", p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
			}
			else if (av_motion_l < 0)
			{
				cv::putText(image, "U", p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
			}

			s << "(" << av_motion_r << ")";
			cv::putText(image, s.str(), p1, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 2);
		}
	}
}

void LKTracker::ShowMotion(cv::Mat& image)
{
	MotionVector::iterator iter;

	int motionR, motionUpDown;
	double motionSumR, motionUpDownSum;

	for(iter = this->regions.begin(); iter != this->regions.end(); ++iter)
	{
		Motion *motion = (*iter);

		motionR = motionUpDown= 0;
		motionSumR = motionUpDownSum = 0;

		cv::Rect origin = motion->getRect();

		cv::rectangle(image, origin, CV_RGB(255, 0, 255), 2);

		for(int i = 0; i < motion->getVx().rows; i++)
		{
			for(int j = 0; j < motion->getVx().cols; j++)
			{
				double x_component = motion->getVx().at<double>(i,j);
				double y_component = motion->getVy().at<double>(i,j);

				//std::cout << "vx " << x_component << " vy " << y_component << std::endl;

				cv::Point p1 = cv::Point(j+origin.x, i+origin.y);
				cv::Point p2 = cv::Point(j+x_component+origin.x, i + y_component+origin.y);

				if(std::abs(y_component) >= 4.0) 
				{
					motionUpDownSum += y_component;
					motionUpDown += 1;
				}

				if(std::abs(y_component) > std::abs(x_component))
					continue;

				if(cv::norm(p1-p2) >= magnitude_treshold)
				{
					// Draw the vector
					cv::circle ( image , p1 , 4 , cv::Scalar(0,255,0) , 2 , 8 );
					cv::line(image, p1, p2, CV_RGB(255, 0, 0), 2);
					cv::circle ( image , p2 , 1 , cv::Scalar(0,255,0) , 2 , 8 );
				}



				if(std::abs(x_component) < 4.0)
					continue;

				motionSumR += x_component;
				motionR++;
			}
		}

		double av_motion_r = motionSumR;
		double av_motion_l = motionUpDownSum / motionUpDown;

		cv::Point p(origin.x+50, origin.y+50);
		cv::Point p1(origin.x+50, origin.y+75);

		std::string text;
		std::ostringstream s;

		if (av_motion_r > magnitude_treshold) {
			cv::putText(image, "L", p, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 3);

			s << "(" << av_motion_r << ")";

			cv::putText(image, s.str(), p1, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 1);
		}
		else if (av_motion_r < -1.0 && std::abs(av_motion_r) >= magnitude_treshold) {
			cv::putText(image, "R", p, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 3);
			
			s << "(" << av_motion_r << ")";

			cv::putText(image, s.str(), p1, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 1);

		}


		p.x+=120;
		p1.x+=120;

		if(av_motion_l == av_motion_l) 
		{
			if (av_motion_l > 5.0) 
			{
				cv::putText(image, "D", p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
			}
			else if (av_motion_l < -5.0)
			{
				cv::putText(image, "U", p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
			}
			else
			{
				continue;
			}

			s << "(" << av_motion_r << ")";
			cv::putText(image, s.str(), p1, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 1);
		}
	}
}

Motion::Motion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next, int windowSize) : windowSize(windowSize)
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
	
	std::cout << "Created region from x: " << x << " y: " << y <<
		" width: " << regionSize.width << " height: " << regionSize.height << std::endl;
	// Create rectangle to extract region later
	this->region = cv::Rect(x, y, regionSize.width, regionSize.height);

	// Initialize derivative
	this->derivative = new Derivative(this->region.height, this->region.width, windowSize);	
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