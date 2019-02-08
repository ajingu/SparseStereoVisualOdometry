#include <opencv2/opencv.hpp>
#include <iostream>

#include "FeatureTracker.h"
#include "FeatureMatcher.h"

using namespace std;
using namespace cv;

#define MAX_FRAME 2 //8

string getFilePath(int frame_number)
{
	return "./images/ZED_image" + to_string(frame_number) +".png";
}

void splitIntoTwo(const Mat src, Mat& left, Mat& right)
{
	int half_width = src.cols / 2;
	int height = src.rows;
	

	left = src(Rect(0, 0, half_width, height));
	right = src(Rect(half_width, 0, half_width, height));
}

int main()
{
	int frame_number = 1;

	//preprocessing
	Mat img_current, img_current_l, img_current_r;
	img_current = imread(getFilePath(frame_number), IMREAD_GRAYSCALE);
    splitIntoTwo(img_current, img_current_l, img_current_r);
	
	//find keypoints from src1
	vector<KeyPoint> kp_current_l, kp_current_r;
	Mat desc_current_l, desc_current_r;
	Ptr<ORB> detector = ORB::create();
	detector->detectAndCompute(img_current_l, noArray(), kp_current_l, desc_current_l);
	detector->detectAndCompute(img_current_r, noArray(), kp_current_r, desc_current_r);

	const float ratio_thresh = 0.8f;
	vector<DMatch> good_matches;
	vector<vector<DMatch>> knn_matches;
	FeatureMatcher featureMacher = FeatureMatcher(ratio_thresh);
	
	featureMacher.matchFeature(desc_current_l, desc_current_r, knn_matches, good_matches);

	//track keypoints between src1 and src2
	Mat img_next, img_next_l, img_next_r;

	vector<Point2f> pts_current_l, pts_next_l;
	vector<uchar> status;
	vector<float> err;

	double focal = 350;
	Point2d pp(336, 336);
	Mat E, R, t, mask;

	vector<KeyPoint> kp_next_l, kp_next_r;
	Mat desc_next_l, desc_next_r;

	Mat dst;

	while (frame_number < MAX_FRAME)
	{
		frame_number++;

		img_next = imread(getFilePath(frame_number), IMREAD_GRAYSCALE);
		splitIntoTwo(img_next, img_next_l, img_next_r);

		KeyPoint::convert(kp_current_l, pts_current_l);
		FeatureTracker tracker = FeatureTracker();
		tracker.trackFeature(img_current_l, img_next_l, pts_current_l, pts_next_l, status, err);

		E = findEssentialMat(pts_next_l, pts_current_l, focal, pp, RANSAC, 0.999, 1.0, mask);
		recoverPose(E, pts_next_l, pts_current_l, R, t, focal, pp, mask);
		cout << R << endl;
		cout << t << endl;

		for (int i = 0; i < pts_next_l.size(); i++)
		{
			kp_next_l.push_back(KeyPoint(pts_next_l[i], 1.f));
		}
		detector->compute(img_next_l, kp_next_l, desc_next_l);
		detector->detectAndCompute(img_next_r, noArray(), kp_next_r, desc_next_r);

		knn_matches.clear();
		good_matches.clear();

		featureMacher.matchFeature(desc_next_l, desc_next_r, knn_matches, good_matches);

		drawMatches(img_next_l, kp_next_l, img_next_r, kp_next_r, good_matches, dst, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


		imshow("image", dst);
		waitKey();
	}

	return 0;
}

