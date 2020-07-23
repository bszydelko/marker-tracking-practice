#include "opencv2/opencv.hpp"
#include <string>
#include "VideoCaptureYUV.h"
#include "processing.h"



int main(int argc, char** argv)
{
	std::string cam = "2";
	std::string sInput = "cam" + cam +"_1920x1080.yuvdist.yuv";
	std::string sPath = "E:\\_SEQ\\" + cam +"\\";
	std::string sFrame = "frame";
	std::string sLightThresh1 = "light thresh 1";
	std::string sLightThersh2 = "light thresh 2";
	std::string sLightMask = "light mask";

	int32_t width = 1920;
	int32_t height = 1080;
	int32_t chromaSubsampling = 420;

	VideoYUV video(sPath + sInput, width, height, chromaSubsampling);
	if (!video.isOpened()) return 0;

	cv::Mat imgFrame;
	cv::Mat imgLightThresh1;
	cv::Mat imgLightThresh2;

	cv::namedWindow(sFrame, cv::WINDOW_KEEPRATIO);
	cv::resizeWindow(sFrame, cv::Size(width / 2, height / 2));

	cv::namedWindow(sLightThresh1, cv::WINDOW_KEEPRATIO);
	cv::resizeWindow(sLightThresh1, cv::Size(width / 3, height / 3));
	cv::namedWindow(sLightThersh2, cv::WINDOW_KEEPRATIO);
	cv::resizeWindow(sLightThersh2, cv::Size(width / 3, height / 3));



	int32_t frame_cnt = 0;

	//thresh1
	video.read(imgFrame);
	frame_cnt++;
	cv::cvtColor(imgFrame, imgFrame, cv::COLOR_YUV2BGR);
	bs::thresholdLights(imgFrame, imgLightThresh1);
	cv::imshow(sLightThresh1, imgLightThresh1);
	//thresh2
	video.read(imgFrame);
	frame_cnt++;
	cv::cvtColor(imgFrame, imgFrame, cv::COLOR_YUV2BGR);
	bs::thresholdLights(imgFrame, imgLightThresh2);
	cv::imshow(sLightThersh2, imgLightThresh2);

	cv::Mat imgLightMask;
	bs::createLightMask(imgLightThresh1, imgLightThresh2, imgLightMask);
	cv::namedWindow(sLightMask, cv::WINDOW_KEEPRATIO);
	cv::resizeWindow(sLightMask, cv::Size(width / 3, height / 3));
	cv::imshow(sLightMask, imgLightMask);


	while (true)
	{
		
		if(cv::waitKey(5) == 27) break;
		if (video.read(imgFrame))
		{
			std::cout << frame_cnt << std::endl;
			frame_cnt++;
			cv::cvtColor(imgFrame, imgFrame, cv::COLOR_YUV2BGR);
			cv::imshow(sFrame, imgFrame);
			cv::waitKey(0);


		}
		else return 0;

	}

	cv::waitKey(0);
	return 0;
}