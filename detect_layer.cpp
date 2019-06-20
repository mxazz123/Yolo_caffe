#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/detect_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

using namespace std;

namespace caffe {
//计算重合部分的边长
template<typename Dtype>
Dtype overlap(Dtype x1, Dtype x2, Dtype w1, Dtype w2){
    Dtype l1 = x1 - w1/2;   //边框1的左边坐标
    Dtype l2 = x2 - w2/2;
    Dtype left = l1 > l2 ? l1 : l2;
    Dtype r1 = x1 + w1/2;
    Dtype r2 = x2 + w2/2;
    Dtype right = r1 > r2 ? r2 : r1;
    return (right - left);
}
//计算边框的IOU，重合部分面积/（两个边跨面积相加-重合部分面积）
template<typename Dtype>
Dtype box_iou(const vector<Dtype> truth, const vector<Dtype> box_out){
    Dtype w = overlap(truth[0],box_out[0],truth[2],box_out[2]);
    Dtype h = overlap(truth[1],box_out[1],truth[3],box_out[3]);
    if (w < 0 || h < 0){return Dtype(0);}
    Dtype inArea = w*h;
    Dtype allArea = truth[2]*truth[3] + box_out[2]*box_out[3] - inArea;
    std::cout << "box_IOU      " << inArea/allArea<< " truth   "  << truth[2]*truth[3]  << "  box_out  " << box_out[2]*box_out[3]  << endl;
    return inArea / allArea;
}

template<typename Dtype>
DetectLayer<Dtype>::DetectLayer(const LayerParameter& param): Layer<Dtype>(param){
    this->layer_param_.add_propagate_down(true);    //bottom[0]为输出值
    this->layer_param_.add_propagate_down(false);   //bottom[1]为标签值
    const DetectParameter& detect_param = this->layer_param_.detect_param();
    classes = detect_param.classes();  //1
    coords = detect_param.coords(); //4
    side = detect_param.side();        //11
    num = detect_param.num();          //默认每个单元格由2个边框
    object_scale = detect_param.object_scale(); //1
    noobject_scale = detect_param.noobject_scale();  //0.5
    class_scale = detect_param.class_scale();   //1
    coord_scale = detect_param.coord_scale();   //5
}

template<typename Dtype>
void DetectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    Layer<Dtype>::LayerSetUp(bottom,top);
    this->layer_param_.add_loss_weight(Dtype(1));
    int label_size = (side*side)*(classes+coords+1);  //726
    int input_size = (side*side)*(classes+num*(1+coords));
    CHECK_EQ(input_size,bottom[0]->count(1)) << "input data Size Error";   //bottom[0]存有(N,C,H,W)的数据，输出一个slice的数目，即（C,H,W）的数量
    CHECK_EQ(label_size,bottom[1]->count(1)) << "label data size error";
}

template<typename Dtype>
void DetectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    vector<int> loss_reshape(0);
    top[0]->Reshape(loss_reshape);
}

//计算loss
template<typename Dtype>
void DetectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    int input_num_each = (side*side)*(classes+num*(1+coords)); //1331
    int truth_num_each = (side*side)*(1+classes+coords); //726
    const Dtype* input = bottom[0]->cpu_data();  //输入数据地址
    const Dtype* truth = bottom[1]->cpu_data();  //label数据地址
    int batch = bottom[0]->num();
    Dtype& cost = top[0]->mutable_cpu_data()[0];   //计算cost,引用
    Dtype* delta = bottom[0]->mutable_cpu_diff();  //指向diff
    cost = Dtype(0.0);

    for(int i = 0; i < bottom[0]->count(); ++i){   //疑问，每个batch有多个slices
        delta[i] = Dtype(0.0);                     //都存在梯度，那么每个梯度是如何综合起来的？
        }

    int location = side * side;

    float avg_iou = 0;
    float avg_cat = 0;
    float avg_allcat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;

    for(int b=0; b < batch; ++b){            //每个batch中的slice
        int input_index = b * input_num_each;    //每个slice的输入数据地址（序列索引）起始位置

        for(int l=0; l < location; ++l){    //slice中的每个网格
            int truth_index = (classes+coords+1)*l + b * truth_num_each;     //当前网格的truth地址
            for (int n=0; n < num; ++n){    //网格中的每个预测框格
            //先计算loss函数中的第４项
            //应该是没有目标物体的网格的置信度误差，此处把所有网格的置信度误差都加上，后面再减去有目标网格的置信度误差
                int confidence_index = input_index + location*classes + l*num + n;
                cost += noobject_scale*pow(input[confidence_index] - 0,2);     //网格没有物品，truth中置信度为0
                delta[confidence_index] = noobject_scale*(input[confidence_index]-0);
                avg_anyobj += input[confidence_index];
            }
            Dtype is_obj = truth[truth_index];       //此网格是否有物体
            if (!is_obj) continue;                   //如果此网格没有物体，则不许进行下列loss计算
            
            //计算loss函数中的第5项
            int class_index = input_index + l*classes;
            for (int j=0; j < classes; ++j){
                cost += class_scale*pow(input[class_index] - truth[truth_index+1+j] ,2);
                delta[class_index + j] = class_scale * (input[class_index+j] - truth[truth_index+1+j]);
                if(truth[truth_index+1+j]) avg_cat += input[class_index+j];
                avg_allcat += input[class_index+j];
            }
            
             //计算loss函数的第1,2项
            vector<float> truthCoords;       //存储truth中的x,y,w,h
            truthCoords.push_back(float(truth[truth_index+1+classes+0])); //x
            truthCoords.push_back(float(truth[truth_index+1+classes+1])); //y
            truthCoords.push_back(float(truth[truth_index+1+classes+2])); //w
            truthCoords.push_back(float(truth[truth_index+1+classes+3])); //h
            
            int n_best;   //存储两个候选框最好的，只使用此候选框进行回归计算
            float best_iou = 0;
            float best_square = 2000;
            for (int n=0; n<num; ++n){
                int inputCoordsIndex = input_index + classes*location + num * location + l * num * coords + n*coords;
                vector<float> inputCoords;    //存储input中的x,y,w,h
                inputCoords.push_back(float(input[inputCoordsIndex + 0])); //x
                inputCoords.push_back(float(input[inputCoordsIndex + 1])); //y
                inputCoords.push_back(float(input[inputCoordsIndex + 2] * input[inputCoordsIndex + 2]));  //压入的是w，但是回归的是w的开方 
                inputCoords.push_back(float(input[inputCoordsIndex + 3] * input[inputCoordsIndex + 3]));  //h
            
                float iou = box_iou(truthCoords, inputCoords);
                //计算input和truth x,y,w,h的平方差，以找出最接近的边框
                float square = pow(inputCoords[0]-truthCoords[0],2)+pow(inputCoords[1]-truthCoords[1],2)+pow(inputCoords[2]-truthCoords[2],2)+pow(inputCoords[3]-truthCoords[3],2);
                
                //找到最接近truth标注的框
                if (iou > 0 || best_iou > 0){
                    if (iou > best_iou) {
                        n_best = n; 
                        best_iou = iou;
                    }
                }
                else{
                    if (square < best_square){
                        n_best = n;
                        best_square = square;
                    }
                }
            }
            //计算x,y,w,h的损失
            int best_coords = input_index + classes*location + num * location + l*num * coords + n_best*coords;
            
            avg_iou += best_iou;
            
            cost += coord_scale*pow(input[best_coords+0]-truth[truth_index+1+classes+0],2) * coord_scale;
            cost += coord_scale*pow(input[best_coords+1]-truth[truth_index+1+classes+1],2) * coord_scale;
            cost += coord_scale*pow(input[best_coords+2]-std::sqrt(truth[truth_index+1+classes+2]),2) * coord_scale;
            cost += coord_scale*pow(input[best_coords+3]-std::sqrt(truth[truth_index+1+classes+3]),2) * coord_scale;
            
            delta[best_coords+0] = coord_scale*(input[best_coords+0] - truth[truth_index+1+classes+0]);
            delta[best_coords+1] = coord_scale*(input[best_coords+1] - truth[truth_index+1+classes+1]);
            delta[best_coords+2] = coord_scale*(input[best_coords+2] - std::sqrt(truth[truth_index+1+classes+2]));
            delta[best_coords+3] = coord_scale*(input[best_coords+3] - std::sqrt(truth[truth_index+1+classes+3]));
            
            //计算loss函数第3项
            //先减去计算第4项lost时多加上的有物体的网格的置信度
            int confidence_index = input_index + location*classes + l*num + n_best;
            cost -= noobject_scale*pow(input[confidence_index] -0, 2);
            cost += object_scale*pow(input[confidence_index] - 1, 2);
            delta[confidence_index] = object_scale*(input[confidence_index]-1);
            avg_obj += input[confidence_index];
            count++;
        }
    }
    printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(classes * count), avg_obj/count, avg_anyobj/(location * batch * num), count);
}

template<typename Dtype>
void DetectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top){
            }

template float overlap(float x1_min, float x1_max, float x2_min, float x2_max);
template double overlap(double x1_min, double x1_max, double x2_min, double x2_max);
template float box_iou(const vector<float> truth, const vector<float> out_box);
template double box_iou(const vector<double> truth, const vector<double> out_box);

}//namespace caffe
