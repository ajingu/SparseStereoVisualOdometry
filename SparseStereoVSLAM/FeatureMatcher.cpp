#include "FeatureMatcher.h"

FeatureMatcher::FeatureMatcher(const float ratio_thresh)
{
	matcher = FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	this->ratio_thresh = ratio_thresh;
}

void FeatureMatcher::matchFeature(Mat& descriptors1, Mat& descriptors2, vector<vector<DMatch>>& knn_matches, vector<DMatch>& good_matches)
{
	matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

	for (int i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i].size() < 2) continue;

		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
}