#include "opencv2/opencv.hpp"
#include "CaptureYUV.h"
#include "MarkerTracker.h"
#include <string>



int main(int argc, char** argv) //TODO - provide support for handling main parameters
{
	std::string sCamNum = "31";
	std::string sSequence = "cam" + sCamNum + "_1920x1080.yuvdist.yuv";
	std::string sMask = "mask_cam" + sCamNum + "_1920x1080.yuvdist.yuv";
	std::string sPath = "E:\\_SEQ\\" + sCamNum +"\\";
	//std::string sPath = "";

	int32_t width = 1920;
	int32_t height = 1080;
	int32_t chromaSubsampling = 420;

	if (argc > 1)
	{
		sPath = "";
		sSequence = argv[1];
		sMask = argv[2];
		width = std::stoi(argv[3]);
		height = std::stoi(argv[4]);
		chromaSubsampling = std::stoi(argv[5]);

	}
	bs::CaptureYUV mask(sPath + sMask, width, height, chromaSubsampling, 1);
	if (!mask.isOpened()) return 0;

	bs::CaptureYUV forwardVideo(sPath + sSequence, width, height, chromaSubsampling, 1);
	if (!forwardVideo.isOpened()) return 0;

	std::cout << "forward: " << std::endl;
	bs::MarkerTracker forwardTracker(&forwardVideo, &mask);
	forwardTracker.start();

	bs::CaptureYUV reverseVideo(sPath + sSequence, width, height, chromaSubsampling, -1);
	if (!forwardVideo.isOpened()) return 0;

	std::cout << "reverse:" << std::endl;
	bs::MarkerTracker reverseTracker(&reverseVideo, &mask);
	reverseTracker.start();

	return 0;
}
