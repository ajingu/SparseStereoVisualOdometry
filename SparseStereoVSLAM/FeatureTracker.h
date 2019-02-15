#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class FeatureTracker
{
public:
	void trackFeature(Mat image1, Mat image2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status);
};