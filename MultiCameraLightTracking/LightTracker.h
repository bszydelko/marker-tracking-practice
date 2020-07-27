#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "VideoCaptureYUV.h"

namespace bs
{
	struct Blob
	{
		cv::Point2f m_position;
		cv::Vec2f m_direction;
		float m_area;
		float m_velocity;

		Blob(cv::Point2f pos);
	};

	class LightTracker
	{
	private:

		//window names
		std::string m_sCurrentFrame	{ "current frame" };
		std::string m_sLightThresh1	{ "light thresh 1" };
		std::string m_sLightThresh2	{ "light htresh 2" };
		std::string m_sLightMask	{ "light mask" };
		std::string m_sBulb			{ "bulb" };

		//frames
		cv::Mat m_imgPreviousFrame;
		cv::Mat m_imgCurrentFrame;
		cv::Mat m_imgLightThresh1;
		cv::Mat m_imgLightThresh2;
		cv::Mat m_imgLightMask;

		bs::VideoCaptureYUV* m_video;
		uint32_t m_frameCounter{ 0 };

		cv::Point m_bulbPos;

		//fun thresholdLights
		cv::Mat m_kernelDilate_thresholdLights;
		int32_t m_thresh_thresholdLights;

		//fun detectLight
		cv::Mat m_imgThresh_detectLight;
		cv::Mat m_imgBulb_detectLight;
		cv::Mat m_kernelDilate_detectLight;
		cv::Mat m_kernelErode_detectLight;

		cv::Mat								m_imgCanny_detectLight;
		std::vector<std::vector<cv::Point>> m_vecContours_detectLight;
		std::vector<cv::Vec4i>				m_vecHierarchy_detectLight;
		std::vector<cv::Moments>			m_vecMoments_detectLight;
		std::vector<cv::Point2f>			m_vecCentralMoments_detectLight;


	public:

		LightTracker(bs::VideoCaptureYUV* video);
		int32_t start();
		~LightTracker();
	protected:
		void thresholdLights(const cv::Mat& frame, cv::Mat& imgThresh);
		void createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask);
		cv::Point2f detectLight(const cv::Mat& frame, const cv::Mat& mask);

	};


	enum STATE
	{
		END_OF_FILE
	};

}