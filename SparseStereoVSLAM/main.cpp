#include <opencv2/opencv.hpp>
#include <iostream>

#include "FeatureTracker.h"
#include "FeatureMatcher.h"

using namespace std;
using namespace cv;

#define MAX_FRAME 3 //8


Point3f calcWordlCoord(Point2f& p_left, Point2f& p_right, float focal , float baseline)
{
	float disparity = p_left.x - p_right.x;

	Point3f p_world;

	p_world.x = p_left.x * baseline / disparity;
	p_world.y = p_left.y * baseline / disparity;
	p_world.z = focal * baseline / disparity;

	return p_world;
}

string getFilePath(int frame_number)
{
	return "./images/ZED_image" + to_string(frame_number) +".png";
}

void splitIntoTwo(const Mat& src, Mat& left, Mat& right)
{
	int half_width = src.cols / 2;
	int height = src.rows;
	

	left = src(Rect(0, 0, half_width, height));
	right = src(Rect(half_width, 0, half_width, height));
}

int main()
{
	Mat img_current, img_current_l, img_current_r, img_next, img_next_l, img_next_r, dst;
	vector<KeyPoint> kp_current_l, kp_current_r, kp_next_l, kp_next_r;
	vector<KeyPoint> kp_current_good_l, kp_current_good_r, kp_next_good_l, kp_next_good_r;
	Mat desc_current_l, desc_current_r, desc_next_l, desc_next_r;
	int frame_number = 1;
	Ptr<ORB> detector = ORB::create();

	vector<DMatch> good_matches_current, good_matches_current_next, good_matches_next;
	vector<vector<DMatch>> knn_matches_current, knn_matches_current_next, knn_matches_next;
	const float ratio_thresh = 0.8f;
	FeatureMatcher matcher = FeatureMatcher(ratio_thresh);

	vector<Point2f> pts_current_l, pts_next_l;
	vector<uchar> status;
	FeatureTracker tracker = FeatureTracker();

	Mat E, R, t, mask;
	const float focal = 350; //pixel
	const Point2d pp(336, 188);
	const float baseline = 0.12f;

	vector<double> distances;
	double absoluteDistance = 0.0;

	//preprocessing
	img_current = imread(getFilePath(frame_number), IMREAD_GRAYSCALE);
    splitIntoTwo(img_current, img_current_l, img_current_r);
	
	//find features from current image
	detector->detectAndCompute(img_current_l, noArray(), kp_current_l, desc_current_l);
	detector->detectAndCompute(img_current_r, noArray(), kp_current_r, desc_current_r);
	

	//match features between current left image and current right image
	matcher.knnMatchStereo(kp_current_l, kp_current_r, desc_current_l, desc_current_r, kp_current_good_l, kp_current_good_r);
	

	while (frame_number < MAX_FRAME)
	{
		frame_number++;

		img_next = imread(getFilePath(frame_number), IMREAD_GRAYSCALE);
		splitIntoTwo(img_next, img_next_l, img_next_r);

		//nextで特徴量計算
		detector->detectAndCompute(img_next_l, noArray(), kp_next_l, desc_next_l);
		detector->detectAndCompute(img_next_r, noArray(), kp_next_r, desc_next_r);

		matcher.knnMatchStereo(kp_next_l, kp_next_r, desc_next_l, desc_next_r, kp_next_good_l, kp_next_good_r);

		//matcherでやったやつをgood_matchesのやつだけ違うvectorにいれればR, tはもとめられる
		//current lとnext lでマッチング
		matcher.extractGoodMatches(desc_current_l, desc_next_l, good_matches_current_next);

		cout << "good_matches: " << good_matches_current_next.size() << endl;
		for (int i = 0; i < good_matches_current_next.size(); i++)
		{
			DMatch match = good_matches_current_next[i];
			pts_current_l.push_back(kp_current_good_l[match.queryIdx].pt);
			pts_next_l.push_back(kp_next_good_l[match.trainIdx].pt);
		}

		//calculate R, t
		E = findEssentialMat(pts_next_l, pts_current_l, focal, pp, RANSAC, 0.999, 1.0, mask);
		recoverPose(E, pts_next_l, pts_current_l, R, t, focal, pp, mask);
		cout << R << endl;
		cout << t << endl;

		/*
		//match features between next left image and next right image
		matcher.knnMatch(kp_next_l, kp_next_r, desc_next_l, desc_next_r, good_matches_next);

		drawMatches(img_next_l, kp_next_l, img_next_r, kp_next_r, good_matches_next, dst, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow("image", dst);
		waitKey();

		//calculate
		distances.clear();
		for (int i = 0; i < good_matches_next.size(); i++)
		{
			DMatch match_next = good_matches_next[i];
			vector<DMatch> matches_current = knn_matches_current[match_next.queryIdx];

			if (matches_current.size() < 2 || matches_current[0].distance > ratio_thresh * matches_current[1].distance) continue;
			DMatch match_current = matches_current[0];

			//current point coordinate
			Point3f p_current = calcWordlCoord(pts_current_l[match_current.queryIdx], kp_current_r[match_current.trainIdx].pt, focal, baseline);
			
			//next point coordinate
			Point3f p_next = calcWordlCoord(kp_next_l[match_next.queryIdx].pt, kp_next_r[match_next.trainIdx].pt, focal, baseline);

			cout << "current z:" << p_current.z << ", next z: " << p_next.z << endl;

			
			distances.emplace_back(norm(p_next-p_current));
		}
		if (distances.size() > 0)
		{
			//output median
			nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
			absoluteDistance = distances[distances.size() / 2];
		}

		cout << absoluteDistance << endl;
		
		waitKey();*/
	}

	return 0;
}

