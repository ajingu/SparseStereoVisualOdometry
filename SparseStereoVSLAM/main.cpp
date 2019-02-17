#include <opencv2/opencv.hpp>
#include <iostream>

#include "FeatureMatcher.h"

using namespace std;
using namespace cv;

#define MAX_FRAME 8 //8


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

	vector<DMatch> good_matches;
	const float ratio_thresh = 0.75f;
	FeatureMatcher matcher = FeatureMatcher(ratio_thresh);

	vector<Point2f> pts_current_l, pts_next_l;

	Mat E, R, t, mask;
	const float focal = 350; //pixel unit
	const Point2d pp(336, 188);
	const float baseline = 0.12f;

	vector<double> distances;
	double absoluteDistance = 0.0;

	//Matching(Current Image)
	img_current = imread(getFilePath(frame_number), IMREAD_GRAYSCALE);
    splitIntoTwo(img_current, img_current_l, img_current_r);
	
	detector->detectAndCompute(img_current_l, noArray(), kp_current_l, desc_current_l);
	detector->detectAndCompute(img_current_r, noArray(), kp_current_r, desc_current_r);

	matcher.knnMatchStereo(kp_current_l, kp_current_r, desc_current_l, desc_current_r, kp_current_good_l, kp_current_good_r);
	

	while (frame_number < MAX_FRAME)
	{
		frame_number++;

		//Matching(Next Image)
		img_next = imread(getFilePath(frame_number), IMREAD_GRAYSCALE);
		splitIntoTwo(img_next, img_next_l, img_next_r);

		detector->detectAndCompute(img_next_l, noArray(), kp_next_l, desc_next_l);
		detector->detectAndCompute(img_next_r, noArray(), kp_next_r, desc_next_r);

		matcher.knnMatchStereo(kp_next_l, kp_next_r, desc_next_l, desc_next_r, kp_next_good_l, kp_next_good_r);

		//Matching(Current Image & Next Image)
		matcher.extractGoodMatches(desc_current_l, desc_next_l, good_matches);
		cout << "Good Matches: " << good_matches.size() << endl;

		//Calculate R, t
		pts_current_l.clear();
		pts_next_l.clear();
		for (int i = 0; i < good_matches.size(); i++)
		{
			DMatch match = good_matches[i];
			pts_current_l.emplace_back(kp_current_good_l[match.queryIdx].pt);
			pts_next_l.emplace_back(kp_next_good_l[match.trainIdx].pt);
		}

		E = findEssentialMat(pts_next_l, pts_current_l, focal, pp, RANSAC, 0.999, 1.0, mask);
		recoverPose(E, pts_next_l, pts_current_l, R, t, focal, pp, mask);
		cout << "Rotation:" << endl << R << endl;
		cout << "Translation;" << endl << t << endl;

		//Calculate absolute distance
		distances.clear();
		for (int i = 0; i < good_matches.size(); i++)
		{
			DMatch match = good_matches[i];
			
			Point3f p_current = calcWordlCoord(kp_current_good_l[match.queryIdx].pt, kp_current_good_r[match.queryIdx].pt, focal, baseline);
			Point3f p_next = calcWordlCoord(kp_next_good_l[match.trainIdx].pt, kp_next_good_r[match.trainIdx].pt, focal, baseline);

			distances.emplace_back(norm(p_next-p_current));
		}

		if (distances.size() > 0)
		{
			nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
			absoluteDistance = distances[distances.size() / 2];
		}

		cout << "Absolute Distance: " << absoluteDistance << endl;

		drawMatches(img_current_l, kp_current_good_l, img_next_l, kp_next_good_l, good_matches, dst, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow("image", dst);
		waitKey();
		
		desc_current_l = desc_next_l;
		kp_current_good_l = move(kp_next_good_l);
		kp_current_good_r = move(kp_next_good_r);
	}

	return 0;
}

