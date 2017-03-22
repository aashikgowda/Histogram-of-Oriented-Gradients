#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
# define PI   3.14159265358979323846
using namespace std;
using namespace cv;
// GLobal variables
#define bin_size 20
#define tot_ang 180
#define cellSize 8

int main(int argc, char* argv[])
{
	
	cv::Mat img, img_pad, img_d;
	int start_i, end_i, start_j, end_j;
	// Read image and convert to Grayscale
	img = imread("image2.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (!img.data)
	{
		cout << "Could not open the image" << endl;
	}
	// Convert image to double
	img.convertTo(img_d, CV_64FC1, 1.0 / 255.0);
	// Padding of image
	const clock_t begin_time = clock();
	cv::copyMakeBorder(img_d, img_pad, 1, 1, 1, 1, BORDER_REPLICATE);
	std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC << "time after image padding-------\n";
	// Initialize gradient and theta variables
	cv::Mat dx = Mat::zeros(img_pad.rows-2, img_pad.cols-2, CV_64FC1);
	cv::Mat dy = Mat::zeros(img_pad.rows-2, img_pad.cols-2, CV_64FC1);
	cv::Mat dxy = Mat::zeros(img_pad.rows-2, img_pad.cols-2, CV_64FC1);
	cv::Mat theta = Mat::zeros(img_pad.rows-2, img_pad.cols-2, CV_64FC1);
	clock_t time1 = clock();
	//img_pad.convertTo(img_dx, CV_16SC1); img_pad.convertTo(img_dy, CV_32FC1); img_pad.convertTo(img_dxy, CV_32FC1);
	// img_dx = img_pad.clone(); img_dy = img_pad.clone(); img_dxy = img_pad.clone();
	// cv::Mat img_dx(img_pad), img_dy(img_pad), img_dxy(img_pad);
	
	// Calculate gradient and theta
	for (int i = 1; i < img_pad.rows - 1; i++)
	{
		for (int j = 1; j < img_pad.cols - 1; j++)
		{
			dx.at<double>(i-1, j-1) = -1 * img_pad.at<double>(i, j - 1) + img_pad.at<double>(i, j + 1);
			dy.at<double>(i-1, j-1) = -1 * img_pad.at<double>(i - 1, j) + img_pad.at<double>(i + 1, j);
			dxy.at<double>(i-1, j-1) = sqrt((dx.at<double>(i-1, j-1)* dx.at<double>(i-1, j-1)) + (dy.at<double>(i-1, j-1)* dy.at<double>(i-1, j-1)));
			theta.at<double>(i-1, j-1) = atan2(dy.at<double>(i-1, j-1), dx.at<double>(i-1, j-1)) * (180 / PI);
			if (theta.at<double>(i-1, j-1) < 0)
				theta.at<double>(i-1, j-1) = theta.at<double>(i-1, j-1) + 180;
		}
   }
	
	std::cout << float(clock() - time1) / CLOCKS_PER_SEC << "time for gradient calculation---------\n";
	
	//cout << theta.at<double>(32, 32) << endl;
	int row = img.rows;
	int cell_counti = floor(row / cellSize);
	int col = img.cols;
	int cell_countj = floor(col / cellSize);
	
	// Initialize a 3-D matrix to store orientation binning
	int size[3] = { cell_counti, cell_countj, 9 };
	cv::Mat orient_bin(3, size, CV_64FC1, cv::Scalar(0));
	
	//double orient_bin[16][8][9] = { { {0} } };
	// for cellI = 1:cellNumI
	clock_t time2 = clock();
	// The following loop in whole performs orientation binning
	for (int cell_i = 0; cell_i < cell_counti; cell_i++)
	{
		//for cellJ = 1 : cellNumJ
		for (int cell_j = 0; cell_j < cell_countj; cell_j++)
		{
			//for bin = 1 : total_Ang / binSize
			for (int bin = 0; bin < 9; bin++)
			{
				// initial iteration within row of cell
				start_i = (cell_i)*cellSize;
				// final iteration within row of cell
				end_i = (cell_i + 1)*cellSize - 1;
				// initial iteration within col of cell
				start_j = (cell_j)*cellSize;
				// final iteration within col of cell
				end_j = (cell_j + 1)*cellSize - 1;

				cv::Mat temp = Mat::zeros(end_i - start_i + 1, end_j - start_j + 1, CV_64FC1);

				//for i = startI:endI
				for (int i = start_i; i <= end_i; i++)
				{
					//for j = startJ : endJ
					for (int j = start_j; j <= end_j; j++)
					{
						// if ((theta(i, j) >= (bin - 1)*binSize + 1) && (theta(i, j)<(bin)*binSize))
						if ((theta.at<double>(i, j) >= (bin)*bin_size + 1) && (theta.at<double>(i, j) < (bin + 1)*bin_size)){
							//	A(i - startI + 1, j - startJ + 1) = 1; 
							temp.at<double>(i - start_i, j - start_j) = 1;
						}
						if (bin > 0){
							if ((theta.at<double>(i, j) >= (bin - 1)*bin_size + 1 + bin_size / 2) && (theta.at<double>(i, j) < (bin)*bin_size)){
								temp.at<double>(i - start_i, j - start_j) = 1 - abs(theta.at<double>(i, j) - ((bin + 1)*bin_size - bin_size / 2)) / bin_size;
							}
						}
						if (bin < tot_ang / bin_size){
							if ((theta.at<double>(i, j) >= (bin + 1)*bin_size + 1) && (theta.at<double>(i, j) < (bin + 2)*bin_size - bin_size / 2)){
								temp.at<double>(i - start_i, j - start_j) = 1 - abs(theta.at<double>(i, j) - ((bin + 1)*bin_size - bin_size / 2)) / bin_size;
							}
						}

					}
				}

				
				orient_bin.at<double>(cell_i, cell_j, bin) = cv::sum(temp.mul(dxy(Range(start_i, end_i + 1), Range(start_j, end_j + 1))))[0];

			}

		}
	}
	std::cout << float(clock() - time2) / CLOCKS_PER_SEC << "time for orientation calculation---------\n";
	
		// cellInBlock = 4;
	int cell_block = 4;
	int block_counti = cell_counti - 1;
	int block_countj = cell_countj - 1;

	//OrientationBinBlocks = zeros(blockNumI, blockNumJ, cellInBlock*total_Ang / binSize);
	int size1[2] = { 1,block_counti*block_countj*36};
	cv::Mat features(2, size1, CV_64FC1, cv::Scalar(0));
	int fstart = 0; 
	int size2[2] = { 1, 9 };
	cv::Mat vect(2, size2, CV_64FC1, cv::Scalar(0));
	clock_t time3 = clock();
	//for blockI = 1:blockNumI
	for (int block_i = 0; block_i < block_counti; block_i++){
		//for blockJ = 1 : blockNumJ
		for (int block_j = 0; block_j < block_countj; block_j++){
			//blockVector = zeros(1, cellInBlock*total_Ang / binSize);
			cv::Mat block_vect = Mat::zeros(1, 36, CV_64FC1);
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					int cell_i = block_i + i;
					int  cell_j = block_j + j;
					//for (int ii = 0; ii < 9; ii++)
					//	vect.at<double>(0, ii) = orient_bin.at<double>(cell_i, cell_j, ii);
					int cell_b_count = 2 * i + j;
					for (int ii = 0; ii < 9; ii++)
					//block_vect(Range(0, 1), Range(cell_b_count * 9, (cell_b_count + 1) * 9)) = vect(Range(0,1),Range(0,9));
					block_vect.at<double>(0, ii + cell_b_count * 9) = orient_bin.at<double>(cell_i, cell_j, ii);
					
				}
			}
		//	cout << block_vect << endl;
			Mat_<float> norm_block_v = (block_vect) / (sum(block_vect)[0]); Mat_<float> norm_block_vect;
		//	cout << norm_block_v << endl;
			threshold(norm_block_v, norm_block_vect, 0.2, 255,THRESH_TRUNC);	
			norm_block_vect = norm_block_vect  / sum(norm_block_vect)[0];
		//	cout << norm_block_vect << endl;
			for (int jj = 0; jj < 36; jj++)
			features.at<double>(0,jj+fstart) = norm_block_vect.at<float>(0,jj);
			fstart += 36;
		}
	}
	 cout << features.at<double>(0, 55) << endl;
	 cout << features.size() << endl;
	 std::cout << float(clock() - time3) / CLOCKS_PER_SEC << "time for bin normalization calculation---------\n";
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", img);
	waitKey(0);
	return 0;
}
