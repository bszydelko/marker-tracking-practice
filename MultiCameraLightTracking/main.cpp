#include "opencv2/opencv.hpp"
#include <string>
#include "VideoCaptureYUV.h"


int main(int argv, char** argc)
{
	std::string sInput = "cam0_1920x1080.yuvdist.yuv";
	std::string sPath = "E:\\_SEQ\\0\\";
	std::string sWindow = "video";


	VideoYUV video(sPath + sInput, 1920, 1080, 420);
	if (!video.isOpened()) return 0;

	cv::Mat frame;

	cv::namedWindow(sWindow, cv::WINDOW_KEEPRATIO);
	cv::resizeWindow(sWindow, cv::Size(1920 / 2, 1080 / 2));

	int frame_cnt = 0;

	while (true)
	{
		
		if(cv::waitKey(5) == 27) break;
		if (video.read(frame))
		{
			std::cout << frame_cnt << std::endl;
			frame_cnt++;
			cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR);
			cv::imshow(sWindow, frame);
		}
		else return 0;
		//waitKey(0);

	}

	cv::waitKey(0);
	return 0;
}