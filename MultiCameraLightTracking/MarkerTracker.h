#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "CaptureYUV.h"
#include <vector>

#define WAIT_TIME 0

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
		MARKER_VISIBLE,
		MARKER_NOT_VISIBLE,
		MARKER_TOO_FAR,
		MARKER_OUT_OF_FRAME,
		MARKER_OVERLAPS_MASK,
	};

	class MarkerTracker
	{
	private:
		std::vector<bs::Marker> m_vecMarker;
		std::vector<cv::Point2d> vecPoints;

		
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

		bs::CaptureYUV* m_video;
		bs::CaptureYUV* m_mask;
		uint32_t m_frameCounter{ 0 };

		cv::Point m_bulbPos;

		//fun thresholdLights
		cv::Mat m_kernelDilate_thresholdLights;
		int32_t m_thresh_thresholdLights{ 210 };

		//fun detectLight
		cv::Mat m_imgThresh;
		cv::Mat m_imgBulb;
		cv::Mat m_kernelDilate;
		cv::Mat m_kernelErode;
		int32_t m_notDetectCount{ 0 };

		cv::Mat								m_imgCanny;
		std::vector<cv::Vec4i>				m_vecContourHierarchy;
		std::vector<cv::Moments>			m_vecMoments;

		int32_t m_frameToReadIdx = 0;
		int32_t m_frameStep = 0;


	public:

		MarkerTracker(bs::CaptureYUV* _video, bs::CaptureYUV* _mask);
		int32_t start();
		~MarkerTracker();

		std::vector<cv::Point2d> getPoints() const;

	protected:
		void threshold_lights(const cv::Mat& frame, cv::Mat& imgThresh);
		void create_light_mask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask);

		MARKER_STATE process_frame(
			const cv::Mat& frame, 
			const cv::Mat& mask,
			std::vector<std::vector<cv::Point>> &vecCurrentContour,
			std::vector<cv::Point2d> &vecCurrentCentralMoments,
			int32_t thresh = 220,
			bool bBlur = true,
			bool bMorph = true);
		cv::Rect2d create_region(const cv::Point2d& predictedMarker, const int32_t expand = 1);
		cv::Point2d select_marker(
			const std::vector<cv::Point2d>& vecRegionCentralMoments, 
			const std::vector<std::vector<cv::Point>>& vecRegionContours,
			const cv::Point2d &predictedMarker, 
			const cv::Rect2d &predictedRegion,
			const cv::Mat& currentMaskRegion);

		bool region_contains_contour(
			const cv::Rect2d& region, 
			const std::vector<std::vector<cv::Point>>& regionContours, 
			const std::vector<std::vector<cv::Point>>& frameContours);

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

}