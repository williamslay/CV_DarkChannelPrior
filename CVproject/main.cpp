//#include "hazeremoval.h"
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main() {
//   Mat in_img;
//   in_img = imread("./img/5.jpeg", IMREAD_UNCHANGED);
//   Mat out_img(in_img.rows, in_img.cols, CV_8UC3);
//   unsigned char* indata = in_img.data;
//   unsigned char* outdata = out_img.data;
//
//   CHazeRemoval hr;
//   cout << hr.InitProc(in_img.cols, in_img.rows, in_img.channels()) << endl;
//   cout << hr.Process(indata, outdata, in_img.cols, in_img.rows, in_img.channels(),3,0.95,0.1,60,0.001) << endl;
//
//   Mat dst;
//   imwrite("./img/result.png",out_img);
//   cv::hconcat(in_img, out_img, dst);
//   namedWindow("IDCP", WINDOW_AUTOSIZE);
//   imshow("IDCP", dst);
//   waitKey(0);
//   return 0;
//}