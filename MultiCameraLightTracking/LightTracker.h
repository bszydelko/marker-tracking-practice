#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "VideoCaptureYUV.h"
#include <vector>

namespace bs
{
	struct Bulb
	{
		cv::Point2d m_position;
		int32_t m_frameNumber;
		bool m_visible;

		cv::Vec2d m_direction;
		double m_velocity;


		Bulb(cv::Point2d& pos, int32_t frameNumber, bool visible = true);
		void setMotion(const bs::Bulb* b);

		friend std::ostream & operator << ( std::ostream &out,const Bulb &b);
	
	};

	class LightTracker
	{
	private:
		std::vector<bs::Bulb> m_vecBulbs;
		
		//window names
		std::string m_sPreviousFrame	{ "previous frame" };
		std::string m_sCurrentFrame		{ "current frame" };
		std::string m_sLightThresh1		{ "light thresh 1" };
		std::string m_sLightThresh2		{ "light htresh 2" };
		std::string m_sLightMask		{ "light mask" };
		std::string m_sBulb				{ "bulb" };
		std::string m_sContoursMoments	{ "contours & central moment" };


		//frames
		cv::Mat m_imgRawFrame;
		cv::Mat m_imgFirstBulbFrame;
		cv::Mat m_imgPreviousFrame;
		cv::Mat m_imgCurrentFrame;
		cv::Mat m_imgLightThresh1;
		cv::Mat m_imgLightThresh2;
		cv::Mat m_imgLightMask;
		cv::Mat m_imgContoursMoments;

		bs::VideoCaptureYUV* m_video;
		uint32_t m_frameCounter{ 0 };

		cv::Point m_bulbPos;

		//fun thresholdLights
		cv::Mat m_kernelDilate_thresholdLights;
		int32_t m_thresh_thresholdLights{ 210 };

		//fun detectLight
		cv::Mat m_imgThresh_detectBulb;
		cv::Mat m_imgBulb_detectBulb;
		cv::Mat m_kernelDilate_detectBulb;
		cv::Mat m_kernelErode_detectBulb;
		int32_t m_notDetectCount{ 0 };

		cv::Mat								m_imgCanny_detectBulb;
		std::vector<std::vector<cv::Point>> m_vecContours_detectBulb;
		std::vector<cv::Vec4i>				m_vecHierarchy_detectBulb;
		std::vector<cv::Moments>			m_vecMoments_detectBulb;
		std::vector<cv::Point2d>			m_vecCentralMoments_detectBulb;
		


	public:

		LightTracker(bs::VideoCaptureYUV* video);
		int32_t start();
		~LightTracker();

	protected:
		void thresholdLights(const cv::Mat& frame, cv::Mat& imgThresh);
		void createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask);

		cv::Point2d detectBulb(const cv::Mat& frame, const cv::Mat& mask);
		cv::Point2d detectBulbInCloseRange(const cv::Mat& frame,const cv::Mat& mask, const cv::Point2d marker);
		cv::Point2d detectBulbInFirstFrame(const cv::Mat& frame, const cv::Mat& mask);

		bool bulbVsMask(const std::vector<cv::Point>& bulbContour, const cv::Mat& mask);
		double distance(const cv::Point2d& p1, const cv::Point2d& p2) const;

		cv::Point2d predictAverage();
		void imshow(
			bool previousFrame,
			bool currentFrame, 
			bool lightThresh1, 
			bool lightThresh2, 
			bool lightMask, 
			bool bulb, 
			bool contoursMoments);


	};


	enum STATE
	{
		END_OF_FILE

	};

	enum BULB_STATE
	{
		BULB_NOT_VISIBLE,
		BULB_TOO_FAR,
		BULB_OUT_OF_FRAME,
		BULB_OVERLAPS_MASK,
	};

}