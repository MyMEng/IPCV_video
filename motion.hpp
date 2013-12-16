#ifndef MOTION_H
#define MOTION_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "derivative.hpp"

// Forward declare classes
class Motion;
class LKTracker;

// Typedefs for vectors

typedef std::vector<Motion*> MotionVector;

// Tracks motion within one specified position
class Motion 
{
private:

	cv::Rect region;
	cv::Mat extractedFrame, extractedNext;

	// Title of windows to display region 
	std::string xWindowTitle, yWindowTitle, tWindowTitle;

	// Hold reference to derivatives for all pixels in the region
	Derivative *derivative;

	void extractRegionAndUpdate(cv::Mat& frame, cv::Mat& next);
public:
	Motion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next);
	~Motion();
	
	void Update(cv::Mat& frame, cv::Mat& next);

	// Get derivative
	cv::Mat getIx() { return this->derivative->getIx(); }
	cv::Mat getIy() { return this->derivative->getIy(); }
	cv::Mat getIt() { return this->derivative->getIt(); }
	cv::Mat getVx() { return this->derivative->getVx(); }
	cv::Mat getVy() { return this->derivative->getVy(); }

	// Get titles of windows to display motion region
	std::string& getWindowTitleX();
	std::string& getWindowTitleY();
	std::string& getWindowTitleT();

	void SetWindowNames(std::string& xWindowTitle, std::string& yWindowTitle, std::string& tWindowTitle);
};

// Tracker for given positions
class LKTracker
{
private:
	MotionVector regions;
public:
	LKTracker();
	~LKTracker();

	void AddRegion(cv::Vec2i position, cv::Size regionSize, cv::Mat& frame, cv::Mat& next);
	void Update(cv::Mat& frame, cv::Mat& next);
	void ShowAll();
	void ShowMotion(cv::Mat& image);
	void detectMotion(cv::Point A, cv::Point B);
};

#endif