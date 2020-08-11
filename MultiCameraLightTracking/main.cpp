#include "opencv2/opencv.hpp"
#include "VideoCaptureYUV.h"
#include "MarkerTracker.h"
#include <string>



int main(int argc, char** argv) //TODO - provide support for handling main parameters
{
	

	std::string sCamNum = "20";
	std::string sFilename = "cam" + sCamNum + "_1920x1080.yuvdist.yuv";
	//std::string sPath = "E:\\_SEQ\\" + sCamNum +"\\";
	std::string sPath = "";

	int32_t width = 1920;
	int32_t height = 1080;
	int32_t chromaSubsampling = 420;

	if (argc > 1)
	{
		sPath = "";
		sFilename = argv[1];
		width = std::stoi(argv[2]);
		height = std::stoi(argv[3]);
		chromaSubsampling = std::stoi(argv[4]);

	}

	bs::VideoCaptureYUV forwardVideo(sPath + sFilename, width, height, chromaSubsampling, 1);
	if (!forwardVideo.isOpened()) return 0;

	std::cout << "forward: " << std::endl;
	bs::MarkerTracker forwardTracker(&forwardVideo);
	forwardTracker.start();

	bs::VideoCaptureYUV reverseVideo(sPath + sFilename, width, height, chromaSubsampling, -1);
	if (!forwardVideo.isOpened()) return 0;

	std::cout << "reverse:" << std::endl;
	bs::MarkerTracker reverseTracker(&reverseVideo);
	reverseTracker.start();




	cv::waitKey(0);
	return 0;
}
