#include "hazeremoval.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;

CHazeRemoval::CHazeRemoval() {
    rows = 0;
    cols = 0;
    channels = 0;
}

CHazeRemoval::~CHazeRemoval() {

}

bool CHazeRemoval::InitProc(int width, int height, int nChannels) {
    bool ret = false;
    rows = height;
    cols = width;
    channels = nChannels;

    if (width > 0 && height > 0 && nChannels == 3) ret = true;
    return ret;
}

bool CHazeRemoval::Process(const unsigned char* indata, unsigned char* outdata, int width, int height,int nChannels,
    int radius, double omega, double t0, int r, double eps) {
    bool ret = true;
    if (!indata || !outdata) {
        ret = false;
    }
    rows = height;
    cols = width;
    channels = nChannels;

    vector<Pixel> tmp_vec;
    Mat p_src = Mat(rows, cols, CV_8UC3, (void*)indata);
    Mat p_dst =  Mat(rows, cols, CV_64FC3);
    Mat p_tran = Mat(rows, cols, CV_64FC1);
    Mat p_gtran = Mat(rows, cols, CV_64FC1);
    Mat p_edst = Mat(rows, cols, CV_64FC3);
    Vec3d* p_Alight = new Vec3d();

#if _MEASURE_RUNTIME_
    auto start0 = std::chrono::high_resolution_clock::now();
    auto start = start0;
#endif
    get_dark_channel(p_src, tmp_vec, rows, cols, channels, radius);
#if _MEASURE_RUNTIME_
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = stop - start;
    std::cout << "get_dark_channel() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
    start = std::chrono::high_resolution_clock::now();
#endif
    get_atmospheric_light(p_src, tmp_vec, p_Alight, rows, cols);
#if _MEASURE_RUNTIME_
    stop = std::chrono::high_resolution_clock::now();
    fp_ms = stop - start;
    std::cout << "get_air_light() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
    start = std::chrono::high_resolution_clock::now();
#endif
    get_transmission(p_src, p_tran, p_Alight, rows, cols, channels, radius , omega);
#if _MEASURE_RUNTIME_
    stop = std::chrono::high_resolution_clock::now();
    fp_ms = stop - start;
    std::cout << "get_transmission() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
    start = std::chrono::high_resolution_clock::now();
#endif
    guided_filter(p_src, p_tran, p_gtran, r, eps);
#if _MEASURE_RUNTIME_
    stop = std::chrono::high_resolution_clock::now();
    fp_ms = stop - start;
    std::cout << "guided_filter() took " << fp_ms.count() << " ms.\n";
#endif

#if _MEASURE_RUNTIME_
    start = std::chrono::high_resolution_clock::now();
#endif
    recover(p_src, p_gtran, p_dst, p_Alight, rows, cols, channels, t0);
#if _MEASURE_RUNTIME_
    stop = std::chrono::high_resolution_clock::now();
    fp_ms = stop - start;
    std::cout << "recover() took " << fp_ms.count() << " ms.\n";
#endif

    exposure(p_dst, p_edst);

#if _MEASURE_RUNTIME_
    start = std::chrono::high_resolution_clock::now();
#endif
    assign_data(outdata, p_edst, rows, cols, channels);
#if _MEASURE_RUNTIME_
    auto stop0 = std::chrono::high_resolution_clock::now();
    stop = stop0;
    fp_ms = stop - start;
    std::cout << "assign_data() took " << fp_ms.count() << " ms.\n";
    fp_ms = stop0 - start0;
    std::cout << "\nTotal runtime: " << fp_ms.count() << " ms.\n";
#endif
    return ret;
}

bool sort_fun(const Pixel& a, const Pixel& b) {
    return a.val > b.val;//descending
}

void get_dark_channel(const cv::Mat& p_src, std::vector<Pixel>& tmp_vec, int rows, int cols, int channels, int radius) {
    int rmin;
    int rmax;
    int cmin;
    int cmax;
    double min_val;
    cv::Vec3b tmp;
    uchar b, g, r;
    std::vector<uchar> tmp_value(3);
    uchar median;
    uchar threshold_lo;
    uchar minpixel;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            rmin = cv::max(0, i - radius);
            rmax = cv::min(i + radius, rows - 1);
            cmin = cv::max(0, j - radius);
            cmax = cv::min(j + radius, cols - 1);
            min_val = 255;
            for (int x = rmin; x <= rmax; x++) {
                for (int y = cmin; y <= cmax; y++) {
                    tmp = p_src.ptr<cv::Vec3b>(x)[y];
                    tmp_value[0] = tmp[0];
                    tmp_value[1] = tmp[1];
                    tmp_value[2] = tmp[2];
                    std::sort(tmp_value.begin(), tmp_value.end());
                    minpixel = tmp_value[0];
                    min_val = cv::min((double)minpixel, min_val);
                }
            }
            tmp_vec.push_back(Pixel(i, j, uchar(min_val)));
        }
    }
    std::sort(tmp_vec.begin(), tmp_vec.end(), sort_fun);
}

void get_atmospheric_light(const cv::Mat& p_src, std::vector<Pixel>& tmp_vec, cv::Vec3d* p_Alight, int rows, int cols) {
    int num = int(rows * cols * 0.001);
    double A_sum[3] = { 0 };
    std::vector<Pixel>::iterator it = tmp_vec.begin();
    for (int cnt = 0; cnt < num; cnt++) {
        cv::Vec3b tmp = p_src.ptr<cv::Vec3b>(it->i)[it->j];
        A_sum[0] += tmp[0];
        A_sum[1] += tmp[1];
        A_sum[2] += tmp[2];
        it++;
    }
    for (int i = 0; i < 3; i++) {
        if (A_sum[i] / num > 220)
            (*p_Alight)[i] = 220;
        else
        (*p_Alight)[i] = A_sum[i] / num;
    }
    cout << "airlight estimated as: " << (*p_Alight)[0] << ", " << (*p_Alight)[1] << ", " << (*p_Alight)[2] << endl;
}

void get_transmission(const cv::Mat& p_src, cv::Mat& p_tran, cv::Vec3d* p_Alight, int rows, int cols, int channels, int radius, double omega) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int rmin = cv::max(0, i - radius);
            int rmax = cv::min(i + radius, rows - 1);
            int cmin = cv::max(0, j - radius);
            int cmax = cv::min(j + radius, cols - 1);
            double min_val = 255.0;
            for (int x = rmin; x <= rmax; x++) {
                for (int y = cmin; y <= cmax; y++) {
                    cv::Vec3b tmp = p_src.ptr<cv::Vec3b>(x)[y];
                    double b = (double)tmp[0] / (*p_Alight)[0];
                    double g = (double)tmp[1] / (*p_Alight)[1];
                    double r = (double)tmp[2] / (*p_Alight)[2];
                    double minpixel = b > g ? ((g > r) ? r : g) : ((b > r) ? r : b);
                    min_val = cv::min(minpixel, min_val);
                }
            }
            p_tran.ptr<double>(i)[j] = 1 - omega * min_val;
        }
    }
}

void guided_filter(cv::Mat& source, cv::Mat& guided_image, cv::Mat& output, int radius, double epsilon)
{
    Mat graymat(source.rows, source.cols, CV_8UC1);
    Mat graymat_64F(source.rows, source.cols, CV_64FC1);
    if (source.type() % 8 != 0)
    {
        cvtColor(source, graymat_64F, COLOR_BGR2GRAY);
    }
    else
    {
        cvtColor(source, graymat, COLOR_BGR2GRAY);
        for (int i = 0; i < source.rows; i++)
        {
            const uchar* inData = graymat.ptr<uchar>(i);
            double* outData = graymat_64F.ptr<double>(i);
            for (int j = 0; j < source.cols; j++)
                *outData++ = *inData++ ;
        }
    }
    //计算I*p和I*I
    Mat mat_Ip(source.rows, source.cols, CV_64FC1),
        mat_I2(source.rows, source.cols, CV_64FC1);
    multiply(graymat_64F, guided_image, mat_Ip);
    multiply(graymat_64F, graymat_64F, mat_I2);

    //计算各种均值
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(guided_image, mean_p, CV_64F, win_size);
    boxFilter(graymat_64F, mean_I, CV_64F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_64F, win_size);
    boxFilter(mat_I2, mean_I2, CV_64F, win_size);

    //计算Ip的协方差和I的方差
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    //求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);
    b = mean_p - a.mul(mean_I);

    //对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64F, win_size);
    boxFilter(b, mean_b, CV_64F, win_size);

    //计算输出 
    Mat output1;
    output = mean_a.mul(graymat_64F) + mean_b;
    imshow("gt", output);
    output.convertTo(output1, CV_8UC3, 255);
    imwrite("./img/gtImage.png", output1);
}

void recover(const cv::Mat& p_src, const cv::Mat& p_gtran, cv::Mat& p_dst, cv::Vec3d* p_Alight, int rows, int cols, int channels, double t0) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int c = 0; c < channels; c++) {
                double val = (double(p_src.ptr<cv::Vec3b>(i)[j][c]) - (*p_Alight)[c]) / cv::max(t0, p_gtran.ptr<double>(i)[j]) + (*p_Alight)[c];
                p_dst.ptr<cv::Vec3d>(i)[j][c] = cv::max(0.0, cv::min(255.0, val));
            }
        }
    }
}

void exposure(const cv::Mat& p_src, cv::Mat& p_edst)
{
    Mat dst = Mat::zeros(p_src.size(), p_src.type());
    double alpha = 1.5; 				   //这里是图像的对比度
    double beta = 15; 					 //这里是图像的亮度
    for (int row = 0; row < p_src.rows; row++) {
        for (int col = 0; col < p_src.cols; col++) {
                double b = p_src.at<Vec3d>(row, col)[0];
                double g = p_src.at<Vec3d>(row, col)[1];
                double r = p_src.at<Vec3d>(row, col)[2];

                p_edst.at<Vec3d>(row, col)[0] = saturate_cast<uchar>(b * alpha + beta);
                p_edst.at<Vec3d>(row, col)[1] = saturate_cast<uchar>(g * alpha + beta);
                p_edst.at<Vec3d>(row, col)[2] = saturate_cast<uchar>(r * alpha + beta);
        }
    }
}

void assign_data(unsigned char* outdata, const cv::Mat& p_dst, int rows, int cols, int channels) {
    for (int i = 0; i < rows * cols * channels; i++) {
        *(outdata + i) = (unsigned char)(*((double*)(p_dst.data) + i));
    }
}