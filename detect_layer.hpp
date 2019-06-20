#ifndef CAFFE_DETECT_LAYER_HPP_
#define CAFFE_DETECT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/detect_layer.hpp"

namespace caffe{

template<typename Dtype>
class DetectLayer : public Layer<Dtype>{
public:
    explicit DetectLayer(const LayerParameter& param);
    virtual ~DetectLayer(){}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const{return "Detect";}
    virtual inline int ExtractNumBottomBlobs() const{return 2;}
    virtual inline int ExtractNumTopBlobs() const {return 1;}

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     //       const vector<Blob<Dtype>*>& top);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


    int classes;
    int coords;
    int side;
    int num;
    float object_scale;
    float noobject_scale;
    float class_scale;
    float coord_scale;
};

INSTANTIATE_CLASS(DetectLayer);
REGISTER_LAYER_CLASS(Detect);
}

#endif