#pragma once

#include <vector>
#include "dl_detect_anchor_box.hpp"

class HumanFaceDetectMNP01 : public dl::detect::DetectAnchorBox<int16_t, int16_t>
{
public:
    /**
     * @brief Construct a new Human Face Detect object
     * 
     * @param score_threshold   The predicted boxes with higher score than the threshold will be remained
     * @param nms_threshold     The predicted boxes with lower IoU than the threshold will be remained
     * @param top_k             The k highest score boxes will be remained 
     */
    HumanFaceDetectMNP01(const float score_threshold, const float nms_threshold, const int top_k);

    /**
     * @brief Destroy the Human Face Detect object
     * 
     */
    ~HumanFaceDetectMNP01();

    /**
     * @brief Forward model and parse output feature map
     * 
     */
    void call();
};