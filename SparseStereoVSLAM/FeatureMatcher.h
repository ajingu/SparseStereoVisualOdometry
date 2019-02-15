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
	void knnMatch(vector<KeyPoint>& kp_l, vector<KeyPoint>& kp_r, Mat& desc_l, Mat& desc_r, vector<vector<DMatch>>& knn_matches, vector<DMatch>& good_matches);
};