#pragma once
#ifndef HAZE_REMOVAL_H
#define HAZE_REMOVAL_H

#include <opencv2\opencv.hpp>
#include <vector>

#define _MEASURE_RUNTIME_ 1

typedef struct _pixel {
	int i;
	int j;
	uchar val;
	_pixel(int _i, int _j, uchar _val) :i(_i), j(_j), val((uchar)_val) {}
} Pixel;

class CHazeRemoval
{
public:
	CHazeRemoval();
	~CHazeRemoval();

public:
	bool InitProc(int width, int height, int nChannels);
	bool Process(const unsigned char* indata, unsigned char* outdata, int width, int height, \
		int nChannels, int radius, double omega, double t0,int r, double eps);

private:
	int rows;
	int cols;
	int channels;

};

void get_dark_channel(const cv::Mat& p_src, std::vector<Pixel>& tmp_vec, int rows, int cols, int channels, int radius);

void get_atmospheric_light(const cv::Mat& p_src, std::vector<Pixel>& tmp_vec, cv::Vec3d* p_Alight, int rows, int cols);

void get_transmission(const cv::Mat& p_src, cv::Mat& p_tran, cv::Vec3d* p_Alight, int rows, int cols, int channels, int radius, double omega);

void guided_filter( cv::Mat& p_src,  cv::Mat& p_tran, cv::Mat& p_gtran, int r, double eps);

void recover(const cv::Mat& p_src, const cv::Mat& p_gtran, cv::Mat& p_dst, cv::Vec3d* p_Alight, int rows, int cols, int channels, double t0);

void exposure(const cv::Mat& p_src, cv::Mat& p_edst);

void assign_data(unsigned char* outdata, const cv::Mat& p_dst, int rows, int cols, int channels);

#endif