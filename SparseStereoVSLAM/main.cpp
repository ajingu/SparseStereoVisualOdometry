#include <opencv2/opencv.hpp>
#include <iostream>

#include "FeatureTracker.h"
#include "FeatureMatcher.h"

using namespace std;
using namespace cv;

void splitIntoTwo(const Mat src, Mat& left, Mat& right)
{
	int half_width = src.cols / 2;
	int height = src.rows;
	

	left = src(Rect(0, 0, half_width, height));
	right = src(Rect(half_width, 0, half_width, height));
}

int main()
{
	//preprocessing
	Mat src1, src2, dst;
	src1 = imread("./images/ZED_image1.png", IMREAD_GRAYSCALE);
	src2 = imread("./images/ZED_image2.png", IMREAD_GRAYSCALE);
	
	Mat src1_l, src1_r, src2_l, src2_r;
	splitIntoTwo(src1, src1_l, src1_r);
	splitIntoTwo(src2, src2_l, src2_r);
	
	//find keypoints from src1
	vector<KeyPoint> kp1_l, kp1_r;
	Mat desc1_l, desc1_r;
	Ptr<ORB> detector = ORB::create();
	detector->detectAndCompute(src1_l, noArray(), kp1_l, desc1_l);
	detector->detectAndCompute(src1_r, noArray(), kp1_r, desc1_r);

	const float ratio_thresh = 0.8f;
	vector<DMatch> good_matches;
	vector<vector<DMatch>> knn_matches;
	FeatureMatcher featureMacher = FeatureMatcher(ratio_thresh);
	
	featureMacher.matchFeature(desc1_l, desc1_r, knn_matches, good_matches);

	//track keypoints between src1 and src2
	vector<Point2f> pts1_l, pts2_l;
	vector<uchar> status;
	vector<float> err;
	KeyPoint::convert(kp1_l, pts1_l);
	FeatureTracker tracker = FeatureTracker();
	tracker.trackFeature(src1_l, src2_l, pts1_l, pts2_l, status, err);
	
	/*
	for (int i = 0; i < pts1_l.size(); i++)
	{
		arrowedLine(src2_l, pts1_l[i], pts2_l[i], Scalar(0, 0, 255));
	}*/

	double focal = 350;
	Point2d pp(336, 336);
	Mat E, R, t, mask;
	E = findEssentialMat(pts1_l, pts2_l, focal, pp, RANSAC, 0.999, 1.0, mask);
	recoverPose(E, pts1_l, pts2_l, R, t, focal, pp, mask);

	cout << R << endl;
	cout << t << endl;

	//find keypoints from src2
	vector<KeyPoint> kp2_l, kp2_r;
	Mat desc2_l, desc2_r;
	for (int i = 0; i < pts2_l.size(); i++)
	{
		kp2_l.push_back(KeyPoint(pts2_l[i], 1.f));
	}
	detector->compute(src2_l, kp2_l, desc2_l);
	detector->detectAndCompute(src2_r, noArray(), kp2_r, desc2_r);

	knn_matches.clear();
	good_matches.clear();

	featureMacher.matchFeature(desc2_l, desc2_r, knn_matches, good_matches);
	
	drawMatches(src2_l, kp2_l, src2_r, kp2_r, good_matches, dst, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	

	imshow("image", dst);
	waitKey();

	return 0;
}

