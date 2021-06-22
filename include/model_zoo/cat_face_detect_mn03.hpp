#pragma once

#include <vector>
#include "dl_detect_anchor_point.hpp"

class CatFaceDetectMN03 : public dl::detect::DetectAnchorPoint<int16_t, int16_t>
{
public:
    /**
     * @brief Construct a new Human Face Detect object
     * 
     * @param score_threshold   The predicted boxes with higher score than the threshold will be remained
     * @param nms_threshold     The predicted boxes with lower IoU than the threshold will be remained
     * @param top_k             The k highest score boxes will be remained 
     * @param resize_scale      The resize scale
     */
    CatFaceDetectMN03(const float score_threshold, const float nms_threshold, const int top_k, const float resize_scale);

    /**
     * @brief Destroy the Cat Face Detect object
     * 
     */
    ~CatFaceDetectMN03();

    /**
     * @brief Forward model and parse output feature map
     * 
     */
    void call();
};
