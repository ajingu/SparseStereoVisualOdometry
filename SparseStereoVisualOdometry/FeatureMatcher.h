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
	void extractGoodMatches(Mat& desc1, Mat& desc2, vector<DMatch>& good_matches);
	void knnMatchStereo(vector<KeyPoint>& kp_l, vector<KeyPoint>& kp_r, Mat& desc_l, Mat& desc_r, vector<KeyPoint>& kp_good_l, vector<KeyPoint>& kp_good_r);
};