#include "FeatureTracker.h"

void FeatureTracker::trackFeature(Mat image1, Mat image2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)
{
	points2.clear();
	status.clear();

	int width = image1.cols;
	int height = image1.rows;
	
	vector<float> err;

	calcOpticalFlowPyrLK(image1, image2, points1, points2, status, err);

	int indexCorrection = 0;
	for (int i = 0; i < status.size(); i++)
	{
		Point2f pt = points2.at(i - indexCorrection);
		if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0) || (pt.x > width-1) || (pt.y > height-1))
		{
			if ((pt.x < 0) || (pt.y < 0) || (pt.x > width - 1) || (pt.y > height - 1)) status.at(i) = 0;
			
			points1.erase(points1.begin() + (i - indexCorrection));
			points2.erase(points2.begin() + (i - indexCorrection));
			indexCorrection++;
		}
	}
}