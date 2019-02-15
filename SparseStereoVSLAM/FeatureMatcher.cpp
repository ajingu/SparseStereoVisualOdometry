#include "FeatureMatcher.h"

FeatureMatcher::FeatureMatcher(const float ratio_thresh)
{
	matcher = FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	this->ratio_thresh = ratio_thresh;
}

void FeatureMatcher::knnMatch(vector<KeyPoint>& kp_l, vector<KeyPoint>& kp_r, Mat& desc_l, Mat& desc_r, vector<vector<DMatch>>& knn_matches, vector<DMatch>& good_matches)
{
	knn_matches.clear();
	good_matches.clear();

	matcher.knnMatch(desc_l, desc_r, knn_matches, 2);

	for (int i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i].size() < 2) continue;
		if (knn_matches[i][0].distance > ratio_thresh * knn_matches[i][1].distance) continue;

		DMatch match = knn_matches[i][0];
		if (kp_l[match.queryIdx].pt.x < kp_r[match.trainIdx].pt.x) continue;

		good_matches.emplace_back(knn_matches[i][0]);
	}

	cout << "knn: " << knn_matches.size() << ", good: " << good_matches.size() << endl;
}