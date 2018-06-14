/*
 * Copyright (c) 2016-2017 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 * Version: 1.07 beta
 * Description:
 * Sample source code showing how to deploy SSD neural network on
 * DeePhi DPU@Zynq7020 platform.
 */
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <sys/types.h>  
#include <sys/wait.h>  
#include <netinet/in.h>  
#include <arpa/inet.h>  
#include <unistd.h>   
#include <signal.h>  
#include <memory.h>  
#include <errno.h>  
#include <stdlib.h>  
#include <time.h>
#include <stdio.h>

// Header file OpenCV for image processing
#include <opencv2/opencv.hpp>
// Header files for DNNDK APIs
#include <dnndk/dputils.h>
#include <dnndk/n2cube.h>

using namespace std;
using namespace cv;
void int2str(const int &int_temp,string &string_temp)  
{  
			stringstream stream;  
			stream<<int_temp;  
			string_temp=stream.str();   //此处也可以用 stream>>string_temp  
} 
int sum_row_min = 0;
		int sum_row_max = 0;
		int avr_row_min;
		int avr_row_max;
		int num_min = 1;
		int num_max = 1;
		int avr=0;
	        int num_save=0;
Mat noparking;
class LaneDetect
{
public:
    Mat currFrame; //stores the upcoming frame,mat means 矩阵
    Mat temp;      //stores intermediate results
    Mat temp2;     //stores the final lane segments

    int diff, diffL, diffR;
    int laneWidth;
    int diffThreshTop;
    int diffThreshLow;
    int ROIrows;
    int vertical_left;//vertical=垂直
    int vertical_right;
    int vertical_top;
    int smallLaneArea;
    int longLane;
    int  vanishingPt;
    float maxLaneWidth;

    //to store various blob properties
    Mat binary_image; //used for blob removal
    int minSize;
    int ratio;
    float  contour_area;
    float blob_angle_deg;
    float bounding_width;
    float bounding_length;
    Size2f sz;//size,2dims,float
    vector< vector<Point> > contours; //浮点类型的轮廓。坐标
    vector<Vec4i> hierarchy; //固定写法，后来findcontours算出边界的坐标，存在contours里面。
    RotatedRect rotated_rect; //填充


    LaneDetect(Mat startFrame)
    {
        //currFrame = startFrame;                                    //if image has to be processed at original size

        currFrame = Mat(320,480,CV_8UC1,0.0);                        //initialised the image size to 320x480  CV_8UC1---则可以创建----8位无符号整形的单通道---灰度图片
        resize(startFrame, currFrame, currFrame.size());             // resize the input to required size

        temp      = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores possible lane markings，rows行，cols列
        temp2     = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores finally selected lane marks

        vanishingPt    = currFrame.rows/2;                           //for simplicity right now //分母变大，增加天空轮廓，分母变小，只认路面
        ROIrows        = currFrame.rows - vanishingPt;               //rows in region of interest
        minSize        = 0.0015 * (currFrame.cols*currFrame.rows);  //min size of any region to be selected as lane //数值为23左右
        maxLaneWidth   = 0.025 * currFrame.cols;                     //approximate max lane width based on image size 宽度为12列左右
        smallLaneArea  = 7 * minSize;
        longLane       = 0.7 * currFrame.rows;
        ratio          = 4;

        //these mark the possible ROI for vertical lane segments and to filter vehicle glare
        vertical_left  = 2*currFrame.cols/5;
        vertical_right = 3*currFrame.cols/5;
        vertical_top   = 2*currFrame.rows/3;

        //namedWindow("lane",3);
        //namedWindow("midstep", 3);
        namedWindow("currframe", 3); //name,数字是窗口size;
        //namedWindow("laneBlobs",3);
		//namedWindow("NoParkingArea",5);

        getLane();
    }

    void getLane()
    {
        //medianBlur(currFrame, currFrame,5 );
        // updateSensitivity();
        //ROI = bottom half
        for(int i=vanishingPt; i<currFrame.rows; i++)
            for(int j=0; j<currFrame.cols; j++)
            {
                temp.at<uchar>(i,j)    = 0;
                temp2.at<uchar>(i,j)   = 0;//temp，temp2的图像，160*480灰度变成0，右半边变成0
            }

        imshow("currframe", currFrame);
        blobRemoval();
    }
	void int2str(const int &int_temp,string &string_temp)  
	{  
			stringstream stream;  
			stream<<int_temp;  
			string_temp=stream.str();   //此处也可以用 stream>>string_temp  
	}  
    void markLane()
    {
        for(int i=vanishingPt; i<currFrame.rows; i++)
        {
            //IF COLOUR IMAGE IS GIVEN then additional check can be done
            // lane markings RGB values will be nearly same to each other(i.e without any hue)

            //min lane width is taken to be 5
            laneWidth =5+ maxLaneWidth*(i-vanishingPt)/ROIrows;
            for(int j=laneWidth; j<currFrame.cols- laneWidth; j++)
            {

                diffL = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j-laneWidth);
                diffR = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j+laneWidth);//（160.5）与（160.0）,（160.10）的差值，，（160,6）与（160,1），（160,11）中间点与两边的差值。。。。。。。多次循环，遍历大多数基本所有像素。
				//总体是计算某点与其水平临近像素方向之差
                diff  =  diffL + diffR - abs(diffL-diffR);//中间点与两边的差值求和，减去差值之差

                //1 right bit shifts to make it 0.5 times
                diffThreshLow = currFrame.at<uchar>(i,j)>>1; //右移一位，除以2，（）下半边都除2）
                //diffThreshTop = 1.2*currFrame.at<uchar>(i,j);

                //both left and right differences can be made to contribute
                //at least by certain threshold (which is >0 right now)
                //total minimum Diff should be atleast more than 5 to avoid noise
                if (diffL>0 && diffR >0 && diff>5)
                    if(diff>=diffThreshLow /*&& diff<= diffThreshTop*/ )
                        temp.at<uchar>(i,j)=255;//根据像素差值找到白线
            }
        }

	
		
		//for (int i = 0; i < 479; i++) 
	//	{
		//	cout<<"uchar="<<temp.at<uchar>(160,i)<<endl;
		//	if (temp.at<uchar>(161, i) >= 200);
			//{
			//	sum_row_min += i;
			//	num_min = num_min + 1;
			//}
			//if (temp.at<uchar>(319, i) >= 200)
			//{
			//	sum_row_max += i;
			//	num_max = num_max + 1;
			//}
			//else
			//{
				//
		//	}
			

	//	}
		//avr_row_min = sum_row_min / num_min;
		//avr_row_max = sum_row_max / num_max;
		//cout << "location of min row is" << avr_row_min << endl;
		//cout << "location of max row is" << avr_row_max << endl;

    }

    void blobRemoval()
    {
	
        markLane();
	static int num=0;
	string name;
	time_t rawtime;
	struct tm *info;
	char buffer[128];
	time(&rawtime);
	info =localtime(&rawtime);
	strftime(buffer,80,"%b%d_%H%M%S",info);
        string ss1=string(buffer);
	//ss1=buffer;
	//cout<<ss1<<endl;
	//std::cout<<ss1<<std::endl;	


        // find all contours in the binary image
        temp.copyTo(binary_image);
        findContours(binary_image, contours,
                     hierarchy, CV_RETR_CCOMP,
                     CV_CHAIN_APPROX_SIMPLE);
		//固定用法//可能需要在这里找到坐标点，https://blog.csdn.net/dcrmg/article/details/51987348//
		//CCOMP检测所有轮廓，但是只建立2个登记关系，外围为顶层，若外围轮廓包含其他轮廓信息，内为轮廓也保存为顶层。
		//SIMPLE仅保存轮廓拐点信息。
        // for removing invalid blobs
        if (!contours.empty()) //有边界就向下执行
        {
            for (size_t i=0; i<contours.size(); ++i)
            {
                //====conditions for removing contours====//

                contour_area = contourArea(contours[i]) ;

                //blob size should not be less than lower threshold
                if(contour_area > minSize)
                {
                    rotated_rect    = minAreaRect(contours[i]);
                    sz              = rotated_rect.size;
                    bounding_width  = sz.width;
                    bounding_length = sz.height;


                    //openCV selects length and width based on their orientation
                    //so angle needs to be adjusted accordingly
                    blob_angle_deg = rotated_rect.angle;
                    if (bounding_width < bounding_length)
                        blob_angle_deg = 90 + blob_angle_deg;   //转换角度，将width与height互换。扁矩形变成长矩形。以X轴为限，逆时针为负角度，顺时针为正角度。所以加90度可以调换矩形方位与宽高

                    //if such big line has been detected then it has to be a (curved or a normal)lane
                    if(bounding_length>longLane || bounding_width >longLane)
                    {
                        drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);  //对drawContours操作
                        drawContours(temp2, contours,i, Scalar(255), CV_FILLED, 8);
                    }

                    //angle of orientation of blob should not be near horizontal or vertical
                    //vertical blobs are allowed only near center-bottom region, where centre lane mark is present
                    //length:width >= ratio for valid line segments
                    //if area is very small then ratio limits are compensated
                    else if ((blob_angle_deg <-10 || blob_angle_deg >-10 ) &&
                             ((blob_angle_deg > -70 && blob_angle_deg < 70 ) ||
                              (rotated_rect.center.y > vertical_top &&
                               rotated_rect.center.x > vertical_left && rotated_rect.center.x < vertical_right)))   //使角度不至于过大或过小，趋近于X或Y轴
                    {

                        if ((bounding_length/bounding_width)>=ratio || (bounding_width/bounding_length)>=ratio
                                ||(contour_area< smallLaneArea &&  ((contour_area/(bounding_width*bounding_length)) > .75) &&
                                   ((bounding_length/bounding_width)>=2 || (bounding_width/bounding_length)>=2)))
                        {
                            drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);
                            drawContours(temp2, contours,i, Scalar(255), CV_FILLED, 8);
							//imshow("123",temp2);
							//waitKey(9999);
                        }
                    }
                }
            }
        }
		//int sum_row_min = 0;
		//int sum_row_max = 0;
		//int avr_row_min;
		//int avr_row_max;
		//int num_min = 1;
		//int num_max = 1;
		//int avr=0;
	        //int num_save=0;
	
		//Mat noparking(320,480,CV_8UC1,128);
		//Mat noparking;
		for (int i = 0; i < currFrame.cols; i++) 
		{
			if (temp.at<uchar>(200, i) >= 200||temp.at<uchar>(210, i) >= 200||temp.at<uchar>(220, i) >= 200||temp.at<uchar>(195, i) >= 200||temp.at<uchar>(190, i) >= 200)
			//{
				//cout<<temp2.at<uchar>(180,i)<<endl;
				//cout<<i<<endl;
				sum_row_min += i;
				num_min = num_min + 1;
			//}
		}
		for(int j=0;j<currFrame.cols;j++)
		{
			//cout<<temp2.at<uchar>(320,j)<<endl;
			if (temp.at<uchar>(320, j) >= 200||temp.at<uchar>(310, j) >= 200||temp.at<uchar>(315, j) >= 200)
			{
				sum_row_max += j;
				num_max = num_max + 1;
				//cout<<max(j)<<endl;
			}

		}
                
	//	Point root_points[1][4];
	//	root_points[0][0]=Point(160,avr_row_min);
	//	root_points[0][1]=Point(320,avr_row_max);
	//	root_points[0][2]=Point(160,480);
	//	root_points[0][3]=Point(320,480);


		
	//	int y_max=avr_row_max;
		//cout<<sum_row_min<<endl;
		//cout<<sum_row_max<<endl;
		//int num_1=0;
		//cout<<sum_row_max<<endl;
		
		avr_row_min = sum_row_min / num_min;
		avr_row_max = sum_row_max / num_max;
		avr=(avr_row_min+avr_row_max)/2;
		cout << "location of min row is" << avr_row_min << endl;
		cout << "location of max row is" << avr_row_max << endl;
		cout<<"avr is "<<avr<<endl;
	//	Rect rect1(170,avr_row_min,450-avr_row_min,140);
	 //   temp2(rect1).copyTo(noparking);
		Rect rect2(avr,160,480-avr,160);
		currFrame(rect2).copyTo(noparking);
		//sprintf(ori,"s%d%s%","ori",++num,".jpg");
		//char ori[128];
		//char nopark[32];
	    //sprintf(nopark,"s%d%s%","nopark",num,".jpg");
		num++;
		//string count=str(num);
		//std::string name = std::to_string(num);
	        int2str(num,name);
                if(num%15==0)
	       {
	        imwrite("/image/"+ss1+".jpg",noparking);
               // imwrite(name+".jpg",noparking);
                imwrite("/image/"+ss1+"ori.jpg",currFrame);
               }
		//if(num%15==1)
		//{
	        //imwrite("/home/sdu/lanedetection/no_parking/"+ss1+"_"+".jpg",noparking);
               // imwrite(name+".jpg",noparking);
                //imwrite("/home/sdu/lanedetection/original/"+ss1+"_"+"ori.jpg",currFrame);
              // }
                //if(num%16==0)
              // {
               // imwrite(name+".jpg",frame);
              // }
                num_save++;
	//waitKey(10000);
		//cout<<num<<endl;
	//	cout<<name<<endl;
		//imshow("NoParkingArea",noparking);
        //imshow("midstep", temp);
        //imshow("laneBlobs", temp2);
       // imshow("lane",currFrame);

	//	nextFrame;


    }


    void nextFrame(Mat &nxt)
    {
        //currFrame = nxt;                        //if processing is to be done at original size

        resize(nxt ,currFrame, currFrame.size()); //resizing the input image for faster processing
        getLane();
    }

    Mat getResult()
    {
        return temp2;
    }

};

// DPU Kernel name for SSD Convolution layers
#define KRENEL_CONV "ssd"
// DPU node name for input and output
#define CONV_INPUT_NODE "conv1_1"
#define CONV_OUTPUT_NODE_LOC "mbox_loc"
#define CONV_OUTPUT_NODE_CONF "mbox_conf"

#define FILE_NAME_LEN  128
#define MAXLINE        1024

// 5.5GOP computation for SSD Convolution layers
const float SSD_WORKLOAD_CONV = 5.5f;

// detection params
const float NMS_THRESHOLD = 0.45;
const float CONF_THRESHOLD = 0.6;
const float PLATE_THRSHOLD = 0.6;
const float PEOPLE_THRSHOLD = 0.6;
//const float MECHANICAL_THRSHOLD = 0.6;
const int TOP_K = 200;
const int KEEP_TOP_K = 100;
const int num_classes = 3;

const string baseImagePath = "/mnt/JPEGImages/";

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;
bool is_running_2 = true;
bool is_displaying = true;

// comparison algorithm for priority_queue
class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

queue<pair<int, Mat>> read_queue;  // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare>
    display_queue;        // display queue
mutex mtx_read_queue;     // mutex of read queue
mutex mtx_display_queue;  // mutex of display queue
int read_index = 0;       // frame index of input video
int display_index = 0;    // frame index to display

/**
 * @brief Calculate softmax on CPU
 *
 * @param src - pointer to int8_t DPU data to be calculated
 * @param size - size of input int8_t DPU data
 * @param scale - scale to miltiply to transform DPU data from int8_t to float
 * @param dst - pointer to float result after softmax
 *
 * @return none
 */
void CPUSoftmax(int8_t* src, int size, float scale, float* dst) {
  float sum = 0.0f;
  for (auto i = 0; i < size; ++i) {
    dst[i] = exp(src[i] * scale);
    sum += dst[i];
  }
  for (auto i = 0; i < size; ++i) {
    dst[i] /= sum;
  }
}
/**
 * @brief Create prior boxes for feature maps of one scale
 *
 * @note Each prior box is represented as a vector: c-x, c-y, width, height,
 * variences.
 *
 * @param image_width - input width of CONV Task
 * @param image_height - input height of CONV Task
 * @param layer_width - width of a feature map
 * @param layer_height - height of a feature map
 * @param variences
 * @param min_sizes
 * @param max_sizes
 * @param aspect_ratios
 * @param offset
 * @param step_width
 * @param step_height
 *
 * @return none
 */
vector<vector<float>> CreateOneScalePriors(
    int image_width, int image_height, int layer_width, int layer_height,
    vector<float>& variances, vector<float>& min_sizes,
    vector<float>& max_sizes, vector<float>& aspect_ratios, float offset,
    float step_width, float step_height) {
  // Generate boxes' dimensions
  vector<pair<float, float>> boxes_dims;
  for (size_t i = 0; i < min_sizes.size(); ++i) {
    // first prior: aspect_ratio = 1, size = min_size
    boxes_dims.emplace_back(min_sizes[i], min_sizes[i]);
    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
    if (!max_sizes.empty()) {
      boxes_dims.emplace_back(sqrt(min_sizes[i] * max_sizes[i]),
                              sqrt(min_sizes[i] * max_sizes[i]));
    }
    // rest of priors
    for (auto ar : aspect_ratios) {
      float w = min_sizes[i] * sqrt(ar);
      float h = min_sizes[i] / sqrt(ar);
      boxes_dims.emplace_back(w, h);
      boxes_dims.emplace_back(h, w);
    }
  }

  vector<vector<float>> priors;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset) * step_width;
      float center_y = (h + offset) * step_height;
      for (auto& dims : boxes_dims) {
        vector<float> box(8);
        // c-x, c-y, width, height
        box[0] = center_x / image_width;
        box[1] = center_y / image_height;
        box[2] = dims.first / image_width;
        box[3] = dims.second / image_height;
        // variances
        for (int i = 4; i < 8; ++i) {
          box[i] = variances[i - 4];
        }
        priors.emplace_back(box);
      }
    }
  }

  return priors;
}

/**
 * @brief Create prior boxes for feature maps of all scales
 *
 * @note Each prior box is represented as a vector: c-x, c-y, width, height,
 * variences.
 *
 * @param none
 *
 * @return all prior boxes arranged from large scales to small scales
 */
vector<vector<float>> CreatePriors() {
  int image_width = 300, image_height = 300;
  int layer_width, layer_height;
  vector<float> variances = {0.1, 0.1, 0.2, 0.2};
  vector<float> min_sizes;
  vector<float> max_sizes;
  vector<float> aspect_ratios;
  float offset = 0.5;
  float step_width, step_height;

  vector<vector<vector<float>>> prior_boxes;
  // conv4_3_norm_mbox_priorbox
  layer_width = 38; 
  layer_height = 38;
  min_sizes = {30.0};
  max_sizes = {60.0};
  aspect_ratios = {2,4};
  step_width = step_height = 8;
  prior_boxes.emplace_back(CreateOneScalePriors(
      image_width, image_height, layer_width, layer_height, variances,
      min_sizes, max_sizes, aspect_ratios, offset, step_width, step_height));
  // fc7_mbox_priorbox
  layer_width = 19;
  layer_height = 19;
  min_sizes = {60.0};
  max_sizes = {111.0};
  aspect_ratios = {2,3,4};
  step_width = step_height = 16;
  prior_boxes.emplace_back(CreateOneScalePriors(
      image_width, image_height, layer_width, layer_height, variances,
      min_sizes, max_sizes, aspect_ratios, offset, step_width, step_height));
  // conv6_2_mbox_priorbox
  layer_width = 10;
  layer_height = 10;
  min_sizes = {111.0};
  max_sizes = {162.0};
  aspect_ratios = {2,3,4};
  step_width = step_height = 32;
  prior_boxes.emplace_back(CreateOneScalePriors(
      image_width, image_height, layer_width, layer_height, variances,
      min_sizes, max_sizes, aspect_ratios, offset, step_width, step_height));
  // conv7_2_mbox_priorbox
  layer_width = 5;
  layer_height = 5;
  min_sizes = {162.0};
  max_sizes = {213.0};
  aspect_ratios = {2, 3,4};
  step_width = step_height = 64;
  prior_boxes.emplace_back(CreateOneScalePriors(
      image_width, image_height, layer_width, layer_height, variances,
      min_sizes, max_sizes, aspect_ratios, offset, step_width, step_height));
  // conv8_2_mbox_priorbox
  layer_width = 3;
  layer_height = 3;
  min_sizes = {213.0};
  max_sizes = {264.0};
  aspect_ratios = {2,4};
  step_width = step_height = 100;
  prior_boxes.emplace_back(CreateOneScalePriors(
      image_width, image_height, layer_width, layer_height, variances,
      min_sizes, max_sizes, aspect_ratios, offset, step_width, step_height));
  // conv9_2_mbox_priorbox
  layer_width = 1;
  layer_height = 1;
  min_sizes = {264.0};
  max_sizes = {315.0};
  aspect_ratios = {2,4};
  step_width = step_height = 300;
  prior_boxes.emplace_back(CreateOneScalePriors(
      image_width, image_height, layer_width, layer_height, variances,
      min_sizes, max_sizes, aspect_ratios, offset, step_width, step_height));

  // prior boxes
  vector<vector<float>> priors;
  int num_priors = 0;
  for (auto& p : prior_boxes) {
    num_priors += p.size();
  }
  priors.clear();
  priors.reserve(num_priors);
  for (size_t i = 0; i < prior_boxes.size(); ++i) {
    priors.insert(priors.end(), prior_boxes[i].begin(), prior_boxes[i].end());
  }
  return priors;
}

/**
 * @brief Calculate overlap ratio of two boxes
 *
 * @note Each box is represented as a vector: left-x, top-y, right-x, bottom-y,
 * label, confidence.
 *
 * @param box1 - reference to a box
 * @param box2 - reference to another box
 *
 * @return ratio of intersection area to union area
 */
float IOU(vector<float>& box1, vector<float>& box2) {
  float left, top, right, bottom;
  float intersection_width, intersection_height, intersection_area, union_area;

  // box type: left-x, top-y, right-x, bottom-y, label, confidence
  left = max(box1[0], box2[0]);
  top = max(box1[1], box2[1]);
  right = min(box1[2], box2[2]);
  bottom = min(box1[3], box2[3]);

  intersection_width = right - left;
  intersection_height = bottom - top;
  if (intersection_width <= 0 || intersection_height <= 0) {
    return 0;
  }

  intersection_area = intersection_width * intersection_height;
  union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) +
               (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area;

  return intersection_area / union_area;
}

/**
 * @brief Non-Maximum Supression
 *
 * @note Each box is represented as a vector: left-x, top-y, right-x, bottom-y,
 * label, confidence.
 *
 * @param boxes - reference to vector of all boxes
 * @param nms_threshold - maximum overlap ratio of two boxes
 *
 * @return vector of kept boxes
 */
vector<vector<float>> NMS(vector<vector<float>>& boxes, float nms_threshold) {
  vector<vector<float>> kept_boxes;
  vector<pair<int, float>> sorted_boxes(boxes.size());
  vector<bool> box_processed(boxes.size(), false);

  for (size_t i = 0; i < boxes.size(); i++) {
    sorted_boxes[i].first = i;
    sorted_boxes[i].second = boxes[i][5];
  }
  sort(sorted_boxes.begin(), sorted_boxes.end(),
       [](const pair<int, float>& ls, const pair<int, float>& rs) {
         return ls.second > rs.second;
       });

  for (size_t pair_i = 0; pair_i < boxes.size(); pair_i++) {
    size_t i = sorted_boxes[pair_i].first;
    if (box_processed[i]) {
      continue;
    }
    kept_boxes.emplace_back(boxes[i]);
    for (size_t pair_j = pair_i + 1; pair_j < boxes.size(); pair_j++) {
      size_t j = sorted_boxes[pair_j].first;
      if (box_processed[j]) {
        continue;
      }
      if (IOU(boxes[i], boxes[j]) >= nms_threshold) {
        box_processed[j] = true;
      }
    }
  }

  return kept_boxes;
}

/**
 * @brief Get final boxes according to location, confidence and prior boxes
 *
 * @note Each box is represented as a vector: left-x, top-y, right-x, bottom-y,
 * label, confidence.
 *       Location, confidence and prior boxes are arranged in the same order to
 * match correspondingly.
 *
 * @param loc - pointer to localization result of all boxes
 * @param loc_scale - scale to miltiply to transform loc from int8_t to float
 * @param conf_softmax - pointer to softmax result of conf
 * @param priors - reference to vector of prior boxes created in initialization
 * @param top_k - maximum number of boxes(with large confidence) to keep
 * @param nms_threshoud - maximum overlap ratio of two boxes
 * @param conf_threshoud - minimum confidence of kept boxes
 *
 * @return vector of final boxes
 */
vector<vector<float>> Detect(int8_t* loc, float loc_scale, float* conf_softmax,
                             vector<vector<float>>& priors, size_t top_k,
                             float nms_threshoud, float conf_threshoud) {
  vector<vector<float>> results;
  vector<vector<float>> boxes;

  auto time1 = chrono::system_clock::now();
  for (size_t i = 0; i < priors.size(); ++i) {
    vector<float> box(6);
    // Compute c-x, c-y, width, height
    float center_x =
        priors[i][2] * loc[i * 4] * loc_scale * priors[i][4] + priors[i][0];
    float center_y =
        priors[i][3] * loc[i * 4 + 1] * loc_scale * priors[i][5] + priors[i][1];
    float width = priors[i][2] * exp(loc[i * 4 + 2] * loc_scale * priors[i][6]);
    float height =
        priors[i][3] * exp(loc[i * 4 + 3] * loc_scale * priors[i][7]);
    // Transform box to left-x, top-y, right-x, bottom-y, label, confidence
    box[0] = center_x - width / 2.0;   // left-x
    box[1] = center_y - height / 2.0;  // top-y
    box[2] = center_x + width / 2.0;   // right-x
    box[3] = center_y + height / 2.0;  // bottom-y
    for (int j = 0; j < 4; ++j) {
      box[j] = min(max(box[j], 0.f), 1.f);
    }
    // Find the maximum confidence and corresponding label
    int label_index = 0;
    float confidence = conf_softmax[i * num_classes];
    for (int j = 1; j < num_classes; ++j) {
      if (conf_softmax[i * num_classes + j] > conf_threshoud) {
        label_index = j;
        confidence = conf_softmax[i * num_classes + j];
      }
    }
    box[4] = label_index;
    box[5] = confidence;
    boxes.emplace_back(box);
  }
  auto time2 = chrono::system_clock::now();

  vector<vector<vector<float>>> detections(num_classes);
  for (size_t i = 0; i < boxes.size(); ++i) {
    int label_index = boxes[i][4];
    detections[label_index].emplace_back(boxes[i]);
  }

  // Apply NMS for each class individually
  vector<vector<float>> boxes_nms;
  for (size_t i = 1; i < detections.size(); ++i) {
    vector<vector<float>> one_class_nms = NMS(detections[i], nms_threshoud);
    for (size_t j = 0; j < one_class_nms.size(); ++j) {
      boxes_nms.emplace_back(one_class_nms[j]);
    }
  }

  auto time3 = chrono::system_clock::now();
  // Get top-k boxes and keep confidence above the threshould
  vector<pair<int, float>> sorted_boxes(boxes_nms.size());
  for (size_t i = 0; i < boxes_nms.size(); ++i) {
    sorted_boxes[i].first = i;
    sorted_boxes[i].second = boxes_nms[i][5];
  }
  sort(sorted_boxes.begin(), sorted_boxes.end(),
       [](const pair<int, float>& ls, const pair<int, float>& rs) {
         return ls.second > rs.second;
       });
  if (top_k > boxes_nms.size()) {
    top_k = boxes_nms.size();
  }
  for (size_t i = 0; i < top_k; ++i) {
    if (boxes_nms[i][5] > conf_threshoud) {
      results.emplace_back(boxes_nms[i]);
    }
  }

  auto time4 = chrono::system_clock::now();
  cout << " prior: " << chrono::duration_cast<chrono::microseconds>(time2-time1).count() << ".us" << endl;
  cout << " nms  : " << chrono::duration_cast<chrono::microseconds>(time3-time2).count() << ".us" << endl;
  cout << " sort : " << chrono::duration_cast<chrono::microseconds>(time4-time3).count() << ".us" << endl;
  return results;
}

int client(char fileName[FILE_NAME_LEN])  
{  
    struct sockaddr_in     serv_addr;  
    char                   buf[MAXLINE];  
    int                    sock_id;  
    int                    read_len;  
    int                    send_len;  
    FILE                   *fp;  
    int                    i_ret;  
        
    /* open the file to be transported commented by  */  
    if ((fp = fopen(fileName,"r")) == NULL) {  
         perror("Open file failed\n");  
         exit(0);  
    }else{

         puts(fileName);
    } 
          
    /* create the socket commented by  */  
    if ((sock_id = socket(AF_INET,SOCK_STREAM,0)) < 0) {  
         perror("Create socket failed\n");  
         exit(0);  
    }  
          
    memset(&serv_addr, 0, sizeof(serv_addr));  
    serv_addr.sin_family = AF_INET;  
    serv_addr.sin_port = htons(atoi("8887"));  
    inet_pton(AF_INET, "192.168.1.100", &serv_addr.sin_addr); 
    //inet_pton(AF_INET, argv[1], &serv_addr.sin_addr);  
         
    /* connect the server commented by  */  
    i_ret = connect(sock_id, (struct sockaddr *)&serv_addr, sizeof(struct sockaddr));  
    if (-1 == i_ret) {  
        printf("Connect socket failed\n");  
        return -1;  
    }  
         
    /* transported the file commented by  */  
    bzero(buf, MAXLINE);  
    while ((read_len = fread(buf, sizeof(char), MAXLINE, fp)) >0 ) {  
        send_len = send(sock_id, buf, read_len, 0);  
        if ( send_len < 0 ) {  
            perror("Send file failed\n");  
            exit(0);  
        }  
        bzero(buf, MAXLINE);  
    }  
      
    fclose(fp);  
    close(sock_id);  
    printf("Send Finish\n");  
    return 0;  
}  
char* getPicName()
{
    time_t rawtime;
    struct tm * timeinfo;
    char *buffer=(char *)malloc(30*sizeof(char));
    if (buffer == 0)
    {
        fprintf(stdout,"can't alloc mem\n");
        return 0;
    }
    else
    {
        memset(buffer,0x00,sizeof(char)*26);
    }

    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (buffer,30,"%F-%I-%M-%S%p.jpg",timeinfo);

    //puts (buffer);

    return buffer;
}  

/**
 * @brief Run DPU and ARM Tasks for SSD, and put image into display queue
 *
 * @param task_conv - pointer to SSD CONV Task
 * @param is_running - status flag of RunSSD thread
 *
 * @return none
 */
void RunSSD(DPUTask* task_conv, Mat& img) {
  // Initializations
  int8_t *loc =
      (int8_t *)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_LOC);
  int8_t *conf =
      (int8_t *)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_CONF);
  float loc_scale = dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_LOC);
//  fprintf(stdout,"loc_scale:%.3f\n",loc_scale);   //shunyi add for debug
  float conf_scale = dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_CONF);
//  fprintf(stdout,"conf_scale:%.3f\n",conf_scale); //shunyi add for debug
  int size = dpuGetOutputTensorSize(task_conv, CONV_OUTPUT_NODE_CONF);
//  int size_loc = dpuGetOutputTensorSize(task_conv, CONV_OUTPUT_NODE_LOC);
//  fprintf(stdout,"mbox_conf size : %d\n",size);   //shunyi add for debug
//  fprintf(stdout,"mbox_loc  size : %d\n",size_loc);   //shunyi add for debug
   
  vector<string> label = {"background","plate","people"};
  vector<vector<float>> priors = CreatePriors();

  float* conf_softmax = new float[size];

//  // Run detection for images in read queue
//    int index;
//    Mat img;
//    mtx_read_queue.lock();
//    if (read_queue.empty()) {
//      mtx_read_queue.unlock();
//      if (is_reading) {
//        continue;
//      } else {
//        is_running = false;
//        break;
//      }
//    } else {
//      index = read_queue.front().first;
//      img = read_queue.front().second;
//      read_queue.pop();
//      mtx_read_queue.unlock();
//    }

    // Set image into CONV Task with mean value
    // dpuSetInputImage(task_conv, (char*)CONV_INPUT_NODE, img, mean);
    dpuSetInputImage2(task_conv, (char *)CONV_INPUT_NODE, img);

    // Run CONV Task on DPU
    auto time1 = chrono::system_clock::now();
    dpuRunTask(task_conv);
    auto time2 = chrono::system_clock::now();
    cout << "dpu time: " << chrono::duration_cast<chrono::microseconds>(time2-time1).count() << ".us" << endl;

    // Show DPU performance every 50 frames
//    if (display_index % 50 == 0) {
//      cout << "Run SSD CONV ..." << endl;
//      // Get DPU execution time (in us) of CONV Task
//      uint64_t timeProf = dpuGetTaskProfile(task_conv);
//      cout << "  DPU CONV Execution time: " << timeProf << "us" << endl;
//      // Get DPU performance for CONV Task
//      cout << "  DPU CONV Performance: "
//           << (SSD_WORKLOAD_CONV * 1.0 / timeProf) * 1000000.0f << "GOPS"
//           << endl;
//    }

    // Run Softmax on ARM since it isn't supported by DPU@Zynq7020
    auto time3 = chrono::system_clock::now();
    for (int i = 0; i < size / num_classes; ++i) {
      CPUSoftmax(&conf[i * num_classes], num_classes, conf_scale,
          &conf_softmax[i * num_classes]);
    }
//    softmax4(conf, size/4, conf_scale, conf_softmax);
    auto time4 = chrono::system_clock::now();

    // Post-process
    vector<vector<float>> results =
        Detect(loc, loc_scale, conf_softmax, priors, TOP_K, NMS_THRESHOLD,
               CONF_THRESHOLD);
    auto time5 = chrono::system_clock::now();
    cout << "softmax : " << chrono::duration_cast<chrono::microseconds>(time4-time3).count() << ".us" << endl;
    cout << "Detect  : " << chrono::duration_cast<chrono::microseconds>(time5-time4).count() << ".us" << endl;
    // char image[128];
    // char image_1[128];
    // int i=0;
    // int j=0;	

    // Modify image to display
    for (size_t i = 0; i < results.size(); ++i) {
      static int count_noparking=0;
      static int count_original=0;     
 
      int label_index = results[i][4];
      float confidence = results[i][5];
      float x_min = results[i][0] * img.cols;
      float y_min = results[i][1] * img.rows;
      float x_max = results[i][2] * img.cols;
      float y_max = results[i][3] * img.rows;
      float width=x_max-x_min;
      float height=y_max-y_min;
      //cout << x_min << "\t" << y_min << "\t" << x_max << "\t" << y_max << "\t"
      //     << results[i][5] << endl;
      if (label_index == 1 && results[i][5] > PLATE_THRSHOLD) {  // car
    time_t rawtime;
	struct tm *info;
	char buffer[128];
	time(&rawtime);
	info =localtime(&rawtime);
	strftime(buffer,80,"%b%d_%H%M",info);
        string ss1=string(buffer);

         rectangle(img, cvPoint(x_min, y_min), cvPoint(x_max, y_max),
                  Scalar(0, 0, 255), 1, 1, 0);
         cout << x_min << "\t" << y_min << "\t" << x_max << "\t" << y_max << "\t"
           << results[i][5] << endl;
         putText(img, label[label_index], cvPoint(x_min, y_min - 2),
                CV_FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 255, 255), 1, 1);
         putText(img, to_string(confidence), cvPoint(x_min, y_min - 12),
                CV_FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 255, 255), 1, 1);
      string temp_ss1=ss1;
      Mat plate;
      Rect rect10(x_min,y_min,width,height);
      img(rect10).copyTo(plate);
      cv::imshow("plate",plate);
      cv::waitKey(10);
      string count_plate="";
      count_original++;
      int2str(count_original,count_plate);
      count_plate="_"+count_plate;
	  ss1+=count_plate;
	  imwrite("/img/"+ss1+".jpg",plate);
	  ss1=temp_ss1;

    //  char fileName[] = "plate.jpg";
      //imwrite(fileName,img);
      //client(fileName);  

      //char * fileName = getPicName();
      //imwrite(fileName,img);
      //client(fileName);


      //if(remove(fileName) != 0){ 
      //    perror("remove file error!");
      //}


      } else if (label_index == 2 && results[i][5] > PEOPLE_THRSHOLD) {  // 
	 time_t rawtime;
	struct tm *info;
	char buffer[128];
	time(&rawtime);
	info =localtime(&rawtime);
	strftime(buffer,80,"%b%d_%H%M",info);
        string ss1=string(buffer);

        rectangle(img, cvPoint(x_min, y_min), cvPoint(x_max, y_max),
                  Scalar(0, 255, 0), 1, 1, 0);
        cout << x_min << "\t" << y_min << "\t" << x_max << "\t" << y_max << "\t"
           << results[i][5] << endl;
        putText(img, label[label_index], cvPoint(x_min, y_min - 2),
                CV_FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 255, 255), 1, 1);
        putText(img, to_string(confidence), cvPoint(x_min, y_min - 12),
                CV_FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 255, 255), 1, 1);
		   string temp_ss1=ss1;
      Mat plate;
      Rect rect10(x_min,y_min,width,height);
      img(rect10).copyTo(plate);
      cv::imshow("plate",plate);
      cv::waitKey(10);
      string count_plate="";
      count_original++;
      int2str(count_original,count_plate);
      count_plate="_"+count_plate;
	  ss1+=count_plate;
	  imwrite("/img/"+ss1+".jpg",plate);
	  ss1=temp_ss1;
        
     //   Mat people;
        //Rect rect1(x_min,y_min,x_max-x_min,y_max-y_min);
        //img(rect1).copyTo(people);
        // sprintf(image_1 ,"/root/s%d%s%","people",++j,".jpg");
      //  char fileName[] = "people.jpg";
	//imwrite(fileName,img);
        //client(fileName);
        //char * fileName = getPicName();
        //imwrite(fileName,img);
        //client(fileName);

        //imshow("people",img);
        //waitKey(1000);
      } //else if (label_index == 3 && results[i][5] > MECHANICAL_THRSHOLD) {  
        //rectangle(img, cvPoint(x_min, y_min), cvPoint(x_max, y_max),
          //        Scalar(255, 0, 0), 1, 1, 0);
         //putText(img, label[label_index], cvPoint(x_min, y_min - 2),
           //     CV_FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 255, 255), 1, 1);
         //putText(img, to_string(confidence), cvPoint(x_min, y_min - 12),
           //     CV_FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 255, 255), 1, 1);
          
        //Mat mechanical;
        //Rect rect1(x_min,y_min,x_max-x_min,y_max-y_min);
        //img(rect1)copyTo(mechanical);
        //imshow("mechanical",mechanical);
        //waitKey(1000);
      //} 
	else {  // background
      }
    }
  
//    display_queue.push(make_pair(index, img));
  delete[] conf_softmax;
  //imshow("check", img);
  //waitKey(200);
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool& is_reading) {
  while (is_reading) {
    Mat img;
    if (read_queue.size() < 30) {
      if (!video.read(img)) {
        cout << "Video end." << endl;
        is_reading = false;
        break;
      }
      mtx_read_queue.lock();
      read_queue.push(make_pair(read_index++, img));
      mtx_read_queue.unlock();
    } else {
      usleep(20);
    }
  }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool& is_displaying) {
  Mat image(360, 480, CV_8UC3);
  imshow("Video Structure @Deephi DPU", image);
  cvMoveWindow("Video Structure @Deephi DPU", 700, 200);
  while (is_displaying) {
    mtx_display_queue.lock();
    if (display_queue.empty()) {
      if (is_running_1 || is_running_2) {
        mtx_display_queue.unlock();
        usleep(20);
      } else {
        is_displaying = false;
        break;
      }
    } else if (display_index == display_queue.top().first) {
      // Display image
      imshow("Video Structure @Deephi DPU", display_queue.top().second);
      display_index++;
      display_queue.pop();
      mtx_display_queue.unlock();
      if (waitKey(1) == 'q') {
        is_reading = false;
        is_running_1 = false;
        is_running_2 = false;
        is_displaying = false;
        break;
      }
    } else {
      mtx_display_queue.unlock();
    }
  }
}

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief Entry for runing SSD neural network
 *
 * @note Neural network SSD has only a DPU kernel as it has no FC layer.
 *       PriorBox layers are replaced by CreatePriors() for optimization
 *       as it needs to be done only once in initialization. Softmax is
 *       run on ARM as it isn't supported by DPU@Zynq7020 platform yet.
 *       (This feature will be enabled on other DPU platforms, and we
 *       already have an ARM realization with neon optimization currently.
 *       Please contact us via dnndk@deephi.tech for more information. )
 *
 * @arg file_name[string] - path to file for detection
 *
 * @return 0 on success, or error message dispalyed in case of failure.
 */
int main(int argc, char** argv) {
  // Check args
  //if (argc != 2) {
  //  cout << "Usage of pic structure demo:./ssd_plate_ppeople file_name[string]"
  //       << endl;
  //  cout << "\tfile_name: path to your file for detection" << endl;
  //  return -1;
  //}

  // DPU Kernels/Tasks for runing SSD
  DPUKernel* kernel_conv;
  DPUTask *task_conv_1, *task_conv_2;

  // Attach to DPU driver and prepare for runing
  dpuOpen();

  // Create DPU Kernels and Tasks for CONV Nodes in SSD
  kernel_conv = dpuLoadKernel(KRENEL_CONV);
  task_conv_1 = dpuCreateTask(kernel_conv, 0);
  task_conv_2 = dpuCreateTask(kernel_conv, 0);

  // Initializations
  //string file_name = argv[1];
  Mat image(720, 1280, CV_8UC3);
  VideoCapture cap(0);
  if(!cap.isOpened())
  {
    return -1;
  }
 
  while(1){

    cap >> image;
    Mat image_bu=image;
    if(image.empty()) break;
    RunSSD(task_conv_1,image);
    LaneDetect detect(image_bu);
    RunSSD(task_conv_1,noparking);
    
  }

  //Mat image = imread(argv[1]);
  //RunSSD(task_conv_1, image);

  // Destroy DPU Tasks and Kernels and free resources
  dpuDestroyTask(task_conv_1);
  dpuDestroyTask(task_conv_2);
  dpuDestroyKernel(kernel_conv);

  // Detach from DPU driver and release resources
  dpuClose();

  //video.release();
  return 0;
}
