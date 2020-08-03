#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "VideoCaptureYUV.h"
#include <vector>

#define WAIT_TIME 0
#define FIRST_FRAME 2 


namespace bs
{
	struct Marker
	{
		cv::Point2d m_position;
		int32_t m_frameNumber;
		bool m_visible;

		cv::Vec2d m_direction;
		double m_velocity;

		Marker() {}
		Marker(cv::Point2d& pos, int32_t frameNumber, bool visible = true);
		void setMotion(const bs::Marker* prevMarker);

		friend std::ostream & operator << ( std::ostream &out,const Marker &b);
	
	};

	enum MARKER_STATE
	{
		BULB_VISIBLE,
		BULB_NOT_VISIBLE,
		BULB_TOO_FAR,
		BULB_OUT_OF_FRAME,
		BULB_OVERLAPS_MASK,
	};

	class MarkerTracker
	{
	private:
		std::vector<bs::Marker> m_vecMarker;
		
		//window names
		std::string m_sPreviousFrame	{ "previous frame" };
		std::string m_sCurrentFrame		{ "current frame" };
		std::string m_sLightThresh1		{ "light thresh 1" };
		std::string m_sLightThresh2		{ "light htresh 2" };
		std::string m_sLightMask		{ "light mask" };
		std::string m_sBulb				{ "bulb" };
		std::string m_sContoursMoments	{ "contours & central moment" };


		//frames
		cv::Mat imgRawFrame;
		cv::Mat imgFirstBulbFrame;
		cv::Mat imgPreviousFrame;
		cv::Mat imgCurrentFrame;
		cv::Mat imgLightThresh1;
		cv::Mat imgLightThresh2;
		cv::Mat imgLightMask;
		cv::Mat imgContoursMoments;

		bs::VideoCaptureYUV* video;
		uint32_t frameCounter{ 0 };

		cv::Point m_bulbPos;

		//fun thresholdLights
		cv::Mat m_kernelDilate_thresholdLights;
		int32_t m_thresh_thresholdLights{ 210 };

		//fun detectLight
		cv::Mat imgThresh;
		cv::Mat imgBulb;
		cv::Mat kernelDilate;
		cv::Mat kernelErode;
		int32_t notDetectCount{ 0 };

		cv::Mat								m_imgCanny;
		std::vector<cv::Vec4i>				m_vecContourHierarchy;
		std::vector<cv::Moments>			m_vecMoments;
		

	public:

		MarkerTracker(bs::VideoCaptureYUV* video);
		int32_t start();
		~MarkerTracker();

	protected:
		void threshold_lights(const cv::Mat& frame, cv::Mat& imgThresh);
		void create_light_mask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask);

		MARKER_STATE process_frame(
			const cv::Mat& frame, 
			const cv::Mat& mask,
			std::vector<std::vector<cv::Point>> &vecCurrentContour,
			std::vector<cv::Point2d> &vecCurrentCentralMoments);
		cv::Rect2d create_region(const cv::Point2d& predictedMarker);
		cv::Point2d select_marker(
			const std::vector<cv::Point2d>& vecCurrentCenterMoments, 
			const std::vector<std::vector<cv::Point>>& vecCurrentContours,
			const cv::Point2d &predictedMarker, 
			const cv::Rect2d &predictedRegion,
			const cv::Mat& currentMaskRegion);

		bool bulbVsMask(const std::vector<cv::Point>& bulbContour, const cv::Mat& mask);
		double distance(const cv::Point2d& p1, const cv::Point2d& p2) const;

		cv::Point2d predict_average();
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

	

}