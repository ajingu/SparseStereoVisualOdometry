#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class FeatureMatcher
{
private:
	FlannBasedMatcher matcher;
	float ratio_thresh;

public:
	FeatureMatcher(const float ratio_thresh);
	void knnMatch(Mat& descriptors1, Mat& descriptors2, vector<vector<DMatch>>& knn_matches, vector<DMatch>& good_matches);
};