#include "FeatureMatcher.h"

FeatureMatcher::FeatureMatcher(const float ratio_thresh)
{
	matcher = FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	this->ratio_thresh = ratio_thresh;
}

void FeatureMatcher::extractGoodMatches(Mat& desc1, Mat& desc2, vector<DMatch>& good_matches)
{
	good_matches.clear();

	vector<vector<DMatch>> knn_matches;
	matcher.knnMatch(desc1, desc2, knn_matches, 2);

	for (int i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.emplace_back(knn_matches[i][0]);
		}
	}
}

void FeatureMatcher::knnMatchStereo(vector<KeyPoint>& kp_l, vector<KeyPoint>& kp_r, Mat& desc_l, Mat& desc_r, vector<KeyPoint>& kp_good_l, vector<KeyPoint>& kp_good_r)
{
	kp_good_l.clear();
	kp_good_r.clear();

	vector<vector<DMatch>> knn_matches;
	matcher.knnMatch(desc_l, desc_r, knn_matches, 2);

	int indexCorrection = 0;

	for (int i = 0; i < knn_matches.size(); i++)
	{
		int j = i - indexCorrection;

		if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			DMatch match = knn_matches[i][0];
			if (kp_l[match.queryIdx].pt.x > kp_r[match.trainIdx].pt.x)
			{
				kp_good_l.emplace_back(kp_l[match.queryIdx]);
				kp_good_r.emplace_back(kp_r[match.trainIdx]);
				continue;
			}
		}

		Rect rect1(0, 0, desc_l.cols, j);
		Rect rect2(0, j + 1, desc_l.cols, desc_l.rows - j - 1);
		Mat newMat;
		newMat.push_back(desc_l(rect1));
		newMat.push_back(desc_l(rect2));
		desc_l = newMat;

		indexCorrection++;
	}
}