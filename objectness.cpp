#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>
#include <sys/types.h>
#include <sys/dir.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace saliency;

int main(int argc, char** argv) {
    Mat img, outimg;
    DIR *d;
    dirent *dp;
    cv::saliency::StaticSaliencySpectralResidual sal;
    cv::saliency::ObjectnessBING bing;
    // calculate saliency for all images in val_images directory.
    char imgin[300];
    strcpy(imgin,argv[1]);
    //strcat(imgin,"/val_images");
    strcat(imgin,"/JPEGImages"); //for pascal.
    Ptr<ObjectnessBING> objectnessBING = makePtr<ObjectnessBING>();
    objectnessBING->setTrainingPath("/home/aseewald/mysrc/opencv_contrib/modules/saliency/samples/ObjectnessTrainedModel");
    objectnessBING->setBBResDir("/home/aseewald/mysrc/opencv_contrib/modules/saliency/samples/ObjectnessTrainedModel/Results");
    d = opendir(imgin);
    std::ofstream outfile;
    char of[300];
    strcpy(of,argv[1]);
    strcat(of,"/objectness/objness.csv");
    outfile.open(of);
    int i = 0;
    printf("Before loop\n");
    while ((dp = readdir(d)) != NULL) {
        if (i < 2) {
            i += 1;
            continue;
        }
        char inname[300];
        strcpy(inname,argv[1]);
        //strcat(inname,"/val_images/");
        strcat(inname,"/JPEGImages/");
        char outname[300];
        strcpy(outname,argv[1]);
        strcat(outname,"/val_saliency/");
        char outname_detect[300];
        strcpy(outname_detect,argv[1]);
        strcat(outname_detect,"/val_boxes/"); 
        strcat(inname,dp->d_name);
        strcat(outname,dp->d_name);
        strcat(outname_detect,dp->d_name);
        printf("%s | %d \n",inname,i);
        img = imread(inname, CV_LOAD_IMAGE_COLOR);
        sal.computeSaliency(img, outimg);
        imwrite(outname, outimg);
        // get boxes.
        std::vector<Vec4i> bb;
        if (objectnessBING->computeSaliency(img,bb)) {
            std::vector<float> objness = objectnessBING->getobjectnessValues();
            assert(objness.size() == bb.size());
            for (int k=0;  k < bb.size(); ++k) {
                if (objness[k] > 500) {
                    continue;
                }
                Vec4i box = bb[k];
                outfile << inname; //first column of csv is filename.
                for(int j = 0; j < 5;++j) {
                    outfile << "," << box[j];
                }
                outfile << objness[k];
                outfile << std::endl;
            }
        }
    ++i;
    }
    outfile.close();
    // calculate objectness for all images in val_images directory.
    closedir(d);
}
