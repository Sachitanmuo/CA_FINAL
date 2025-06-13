#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

float computeLuminance(const Vec3f& color) {
    return 0.2126f * color[2] + 0.7152f * color[1] + 0.0722f * color[0];
}

void computeLogLuminance(const Mat& img, Mat& logLum) {
    logLum.create(img.rows, img.cols, CV_32F);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            logLum.at<float>(y, x) = logf(1.0f + computeLuminance(img.at<Vec3f>(y, x)));
}

void computeGradients(const Mat& logLum, Mat& gx, Mat& gy) {
    gx = Mat::zeros(logLum.size(), CV_32F);
    gy = Mat::zeros(logLum.size(), CV_32F);
    for (int y = 0; y < logLum.rows - 1; ++y)
        for (int x = 0; x < logLum.cols - 1; ++x) {
            gx.at<float>(y, x) = logLum.at<float>(y, x + 1) - logLum.at<float>(y, x);
            gy.at<float>(y, x) = logLum.at<float>(y + 1, x) - logLum.at<float>(y, x);
        }
}

void compressGradients(Mat& gx, Mat& gy, float alpha) {
    for (int y = 0; y < gx.rows; ++y)
        for (int x = 0; x < gx.cols; ++x) {
            gx.at<float>(y, x) /= (1.0f + alpha * fabsf(gx.at<float>(y, x)));
            gy.at<float>(y, x) /= (1.0f + alpha * fabsf(gy.at<float>(y, x)));
        }
}

void computeDivergence(const Mat& gx, const Mat& gy, Mat& div) {
    div = Mat::zeros(gx.size(), CV_32F);
    for (int y = 1; y < gx.rows; ++y)
        for (int x = 1; x < gx.cols; ++x)
            div.at<float>(y, x) = (gx.at<float>(y, x) - gx.at<float>(y, x - 1)) +
                                  (gy.at<float>(y, x) - gy.at<float>(y - 1, x));
}

void poissonJacobi(Mat& u, const Mat& div, int iterations) {
    Mat uNew = u.clone();
    for (int k = 0; k < iterations; ++k) {
        for (int y = 1; y < u.rows - 1; ++y)
            for (int x = 1; x < u.cols - 1; ++x) {
                uNew.at<float>(y, x) = 0.25f * (
                    u.at<float>(y - 1, x) + u.at<float>(y + 1, x) +
                    u.at<float>(y, x - 1) + u.at<float>(y, x + 1) -
                    div.at<float>(y, x));
            }
        uNew.copyTo(u);
    }
}

Mat toneMapMantiuk(const Mat& input, float alpha, float offset, float gammaDen, int iterations) {
    Mat img;
    input.convertTo(img, CV_32FC3, 1.0 / 255.0);
    Mat logLum, gx, gy, div;
    computeLogLuminance(img, logLum);
    computeGradients(logLum, gx, gy);
    compressGradients(gx, gy, alpha);
    computeDivergence(gx, gy, div);

    Mat u = logLum.clone();
    poissonJacobi(u, div, iterations);

    float gamma = 1.0f / gammaDen;
    Mat output(img.size(), CV_8UC3);
    const float EPS = 1e-4f;
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            float mapped = expf(u.at<float>(y, x)) - 1.0f + offset;
            Vec3f rgb = img.at<Vec3f>(y, x);
            float origLum = computeLuminance(rgb) + EPS;
            float scale = mapped / origLum;
            Vec3f newRGB = Vec3f(
                powf(rgb[0] * scale, gamma),
                powf(rgb[1] * scale, gamma),
                powf(rgb[2] * scale, gamma));
            output.at<Vec3b>(y, x) = Vec3b(
                min(255.f, newRGB[0] * 255.0f),
                min(255.f, newRGB[1] * 255.0f),
                min(255.f, newRGB[2] * 255.0f));
        }
    }
    return output;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./mantiuk_cpu input.png output.png [alpha offset gammaDen iters]\n";
        return -1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    float alpha = (argc > 3) ? atof(argv[3]) : 0.2f;
    float offset = (argc > 4) ? atof(argv[4]) : 0.01f;
    float gammaDen = (argc > 5) ? atof(argv[5]) : 2.2f;
    int iterations = (argc > 6) ? atoi(argv[6]) : 500;

    Mat input = imread(inputFile, IMREAD_COLOR);
    if (input.empty()) {
        cerr << "Error reading image: " << inputFile << endl;
        return -1;
    }

    Mat result = toneMapMantiuk(input, alpha, offset, gammaDen, iterations);
    imwrite(outputFile, result);
    cout << "Done." << endl;
    return 0;
}
