#include "opencv2/opencv.hpp"
#include "VideoCaptureYUV.h"
#include "LightTracker.h"
#include <string>



int main(int argc, char** argv) //TODO - provide support for handling main parameters
{

	std::string sCamNum = "0";
	std::string sFilename = "cam" + sCamNum + "_1920x1080.yuvdist.yuv";
	//std::string sPath = "E:\\_SEQ\\" + cam +"\\";
	std::string sPath = "";

	int32_t width = 1920;
	int32_t height = 1080;
	int32_t chromaSubsampling = 420;

	bs::VideoCaptureYUV video(sPath + sFilename, width, height, chromaSubsampling);
	if (!video.isOpened()) return 0;

	bs::LightTracker tracker(&video);
	
	tracker.start();


	cv::waitKey(0);
	return 0;
}
