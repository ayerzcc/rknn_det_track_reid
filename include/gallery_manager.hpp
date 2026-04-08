/**
 * @file gallery_manager.hpp
 * @brief GalleryManager类头文件 - 定义特征库管理器的接口
 * @details 支持1024维特征向量存储和管理，用于人员重识别
 */

#ifndef GALLERY_MANAGER_HPP
#define GALLERY_MANAGER_HPP

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <map>

#include <opencv2/opencv.hpp>

/**
 * @class GalleryManager
 * @brief 特征库管理器类，负责特征向量的存储、加载和管理
 * @note 支持1024维特征向量，优化人员重识别性能
 */
class GalleryManager 
{

public:
    /**
     * @brief 图库目标信息结构体
     */
    struct GalleryTarget {
        std::string id;           ///< 目标ID
        std::string dir;          ///< 图库目录
        // cv::Mat thumbnail;     ///< 缩略图（待实现）
    };

    /**
     * @brief 构造函数
     */
    GalleryManager();

    /**
     * @brief 加载特征图库
     */
    void load_gallery();

    /**
     * @brief 加载选中的特征图库
     * @param target_id 目标ID
     */
    void load_selected_gallery(const std::string& target_id);

    /**
     * @brief 保存特征和截图到图库
     * @param feature 特征向量（1024维）
     * @param frame 图像帧
     * @param target_id 目标ID
     */
    void save_to_gallery(const std::vector<float>& feature, const cv::Mat& frame, const std::string& target_id);

    /**
     * @brief 计算平均相似度
     * @param current_feature 当前特征向量
     * @return 平均相似度分数
     */
    float get_avg_similarity(const std::vector<float>& current_feature) const;

    /**
     * @brief 计算最大相似度
     * @param current_feature 当前特征向量
     * @return 最大相似度分数
     */
    float get_max_similarity(const std::vector<float>& current_feature) const;

    /**
     * @brief 重置特征图库
     * @param target_id 目标ID
     */
    void reset_gallery(const std::string& target_id);

    /**
     * @brief 检查图库是否已加载
     * @return 已加载返回true，否则返回false
     */
    bool is_gallery_loaded() const;

    /**
     * @brief 获取图库目标列表
     * @return 图库目标列表
     */
    std::vector<GalleryTarget> get_gallery_targets() const;

    /**
     * @brief 获取当前加载的目标ID
     * @return 目标ID
     */
    std::string get_loaded_target_id() const;

    bool is_gallery_full(void);

private:
    /**
     * @brief 确保特征向量兼容性
     * @param feature 输入特征向量
     * @return 兼容的特征向量
     */
    std::vector<float> ensure_feature_compatibility(const std::vector<float>& feature) const;

    /**
     * @brief 标准化特征向量
     * @param feature 输入特征向量
     * @return 标准化后的特征向量
     */
    std::vector<float> normalize_feature(const std::vector<float>& feature) const;

    /**
     * @brief 创建目录（如果不存在）
     * @param path 目录路径
     */
    void create_directory_if_not_exists(const std::string& path) const;


private:
    std::string gallery_dir_;                          ///< 图库目录
    std::vector<std::vector<float>> gallery_features_; ///< 特征向量列表（1024维）
    std::vector<GalleryTarget> gallery_targets_;       ///< 图库目标列表
    std::string target_gallery_dir_;                   ///< 当前目标图库目录
    std::string loaded_target_id_;                     ///< 已加载的目标ID
    bool gallery_loaded_;                              ///< 图库是否已加载    

    int m_iFeatureVectorLen = 1024;
    int m_iMaxGallerySize = 20;
    int m_iIsShow = 0;
    int m_iUpdateIndex = 7;

};

#endif // GALLERY_MANAGER_HPP
