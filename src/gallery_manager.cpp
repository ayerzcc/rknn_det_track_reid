/**
 * @file gallery_manager.cpp
 * @brief GalleryManager类的实现文件 - 特征库管理器实现
 * @details 支持1024维特征向量存储和管理，用于人员重识别
 */

#include "gallery_manager.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "config.hpp"

/**
 * @brief 构造函数
 */
GalleryManager::GalleryManager()
{
    std::shared_ptr<Config> params_config = Config::getDefaultInstance();
    gallery_loaded_ = false;
    m_iIsShow = params_config->getKeyValue<int>("debug_show");
    m_iFeatureVectorLen = params_config->getKeyValue<int>("gallery_feature_length");
    m_iMaxGallerySize = params_config->getKeyValue<int>("gallery_max_size");
    gallery_dir_ = params_config->getKeyValue<std::string>("gallerry_database_dir");

    m_iUpdateIndex = m_iMaxGallerySize / 2;

    // 创建图库目录
    // create_directory_if_not_exists(gallery_dir_);
    // if(m_iIsShow >= 1){
    //     std::cout << "特征图库目录: " << gallery_dir_ << std::endl;
    // }
    
    // 加载已有特征库
    load_gallery();

    return ;
}

bool GalleryManager::is_gallery_full(void){

    if (gallery_features_.size() >= m_iMaxGallerySize){
        return true;
    }

    return false;
}

/**
 * @brief 加载特征图库
 */
void GalleryManager::load_gallery() {
    // 检查图库目录是否存在
    // if (!std::filesystem::exists(gallery_dir_)) {
    //     if(m_iIsShow >= 0){
    //         std::cout << "特征图库目录不存在" << std::endl;
    //     }
    //     gallery_loaded_ = false;
    //     return;
    // }
    
    // // 查找所有目标图库
    // std::vector<GalleryTarget> targets;
    // for (const auto& entry : std::filesystem::directory_iterator(gallery_dir_)) {
    //     if (entry.is_directory()) {
    //         std::string dir_name = entry.path().filename().string();
    //         if (dir_name.find("target_") == 0) {
    //             std::string target_id = dir_name.substr(7); // 去掉"target_"前缀
                
    //             // 检查是否有特征文件
    //             bool has_features = false;
    //             for (const auto& file_entry : std::filesystem::directory_iterator(entry.path())) {
    //                 if (file_entry.path().extension() == ".npy") {
    //                     has_features = true;
    //                     break;
    //                 }
    //             }
                
    //             if (has_features) {
    //                 GalleryTarget target;
    //                 target.id = target_id;
    //                 target.dir = entry.path().string();
    //                 targets.push_back(target);
    //             }
    //         }
    //     }
    // }
    
    // gallery_targets_ = targets;
    // if(m_iIsShow >= 1){
    //     std::cout << "找到 " << gallery_targets_.size() << " 个特征图库目标" << std::endl;
    // }
        
    // gallery_loaded_ = !gallery_targets_.empty();

    return ;
}

/**
 * @brief 加载选中的特征图库
 * @param target_id 目标ID
 */
void GalleryManager::load_selected_gallery(const std::string& target_id) {
    // 查找目标图库
    for (const auto& target : gallery_targets_) {
        if (target.id == target_id) {
            target_gallery_dir_ = target.dir;
            loaded_target_id_ = target_id;
            
            // 加载特征向量
            gallery_features_.clear();
            for (const auto& entry : std::filesystem::directory_iterator(target_gallery_dir_)) {
                if (entry.path().extension() == ".npy") {
                    // 读取特征文件（简化实现，实际需要二进制读取）
                    std::ifstream file(entry.path(), std::ios::binary);
                    if (file) {
                        // 获取文件大小
                        file.seekg(0, std::ios::end);
                        size_t size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        
                        // 检查特征维度
                        if (size == m_iFeatureVectorLen * sizeof(float)) {
                            std::vector<float> feature(m_iFeatureVectorLen);
                            file.read(reinterpret_cast<char*>(feature.data()), size);
                            gallery_features_.push_back(feature);
                        }
                    }
                }
            }
            if(m_iIsShow >= 1){
                std::cout << "成功加载 " << gallery_features_.size() << " 个特征向量" << std::endl;
            }
            gallery_loaded_ = true;
            return;
        }
    }

    if(m_iIsShow >= 0){
        std::cout << "未找到目标 " << target_id << " 的特征图库" << std::endl;
    }
        
    return ;
}

/**
 * @brief 保存特征和截图到图库
 * @param feature 特征向量（512维）
 * @param frame 图像帧
 * @param target_id 目标ID
 */
void GalleryManager::save_to_gallery(const std::vector<float>& feature, const cv::Mat& frame, const std::string& target_id) {
    if (target_id.empty() || feature.empty() || frame.empty()) {
        if(m_iIsShow >= 0){
            std::cerr << "save_to_gallery error: " << "invalid input paramters!" << std::endl;
        }
        return;
    }

    // if (gallery_features_.size() >= m_iMaxGallerySize) {
    //     return; 
    // }
    
    // // 确保图库目录存在
    // std::string target_gallery_dir = gallery_dir_ + "/target_" + target_id;
    // create_directory_if_not_exists(target_gallery_dir);
    
    // 确保特征兼容性
    std::vector<float> compatible_feature = ensure_feature_compatibility(feature);
    if (compatible_feature.empty()) {
        if(m_iIsShow >= 0){
            std::cerr << "特征向量不兼容，保存失败" << std::endl;
        }
        return;
    }
    
    // 生成唯一文件名
    // auto now = std::chrono::system_clock::now();
    // auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    // std::string base_filename = std::to_string(timestamp) + "_" + std::to_string(gallery_features_.size());
    
    // 保存特征向量
    // std::string npy_file = target_gallery_dir + "/" + base_filename + ".npy";
    // std::ofstream file(npy_file, std::ios::binary);
    // if (file) {
        // file.write(reinterpret_cast<const char*>(compatible_feature.data()), compatible_feature.size() * sizeof(float));
        
    // }
    
    // 保存图像（简化实现，实际需要编码保存）
    // std::string img_file = target_gallery_dir + "/" + base_filename + ".jpg";
    // cv::imwrite(img_file, frame); // 需要OpenCV支持

    // int ret = system("sync;sync;sync;");
    // if(ret != 0){
    //     if(m_iIsShow >= 0){
    //         std::cerr << "syste(\"sync\") execute error!" << std::endl;
    //     }
    // }
    
    // 限制图库大小
    if (gallery_features_.size() >= m_iMaxGallerySize) {
        //gallery_features_.erase(gallery_features_.begin());
        // 删除最旧的文件（简化实现）
        // std::vector<std::string> files_to_delete;
        // for (const auto& entry : std::filesystem::directory_iterator(target_gallery_dir)) {
        //     files_to_delete.push_back(entry.path().string());
        // }
        // sort(files_to_delete.begin(), files_to_delete.end());
        // if (files_to_delete.size() >= 2) {
        //     for (size_t i = 0; i < 2; i++) {
        //         std::filesystem::remove(files_to_delete[i]);
        //     }
        // }

        gallery_features_[m_iUpdateIndex] = compatible_feature;
        m_iUpdateIndex++;
        if(m_iUpdateIndex >= m_iMaxGallerySize){
            m_iUpdateIndex = m_iMaxGallerySize / 2;
        }

    }else{
        gallery_features_.push_back(compatible_feature);
    }

    return ;
}

/**
 * @brief 计算平均相似度
 * @param current_feature 当前特征向量
 * @return 平均相似度分数
 */
float GalleryManager::get_avg_similarity(const std::vector<float>& current_feature) const 
{
    if (gallery_features_.empty() || current_feature.empty()) {
        return 0.0f;
    }
    
    std::vector<float> similarities;
    std::vector<float> normalized_current = normalize_feature(current_feature);
    
    for (const auto& feature : gallery_features_) {
        std::vector<float> normalized_feature = normalize_feature(feature);
        
        // 计算点积
        float dot_product = inner_product(
            normalized_current.begin(), normalized_current.end(),
            normalized_feature.begin(), 0.0f);
        
        // 计算模长
        float norm_current = sqrt(inner_product(
            normalized_current.begin(), normalized_current.end(),
            normalized_current.begin(), 0.0f));
        
        float norm_feature = sqrt(inner_product(
            normalized_feature.begin(), normalized_feature.end(),
            normalized_feature.begin(), 0.0f));
        
        // 计算余弦相似度
        float similarity = dot_product / (norm_current * norm_feature + 1e-8f);
        similarities.push_back(similarity);
    }
    
    // 计算平均相似度
    float avg_similarity = accumulate(similarities.begin(), similarities.end(), 0.0f) / similarities.size();
    return avg_similarity;
}

/**
 * @brief 计算最大相似度
 * @param current_feature 当前特征向量
 * @return 最大相似度分数
 */
float GalleryManager::get_max_similarity(const std::vector<float>& current_feature) const 
{
    if (gallery_features_.empty() || current_feature.empty()) {
        return 0.0f;
    }
    
    std::vector<float> normalized_current = normalize_feature(current_feature);
    float max_sililarity = 0.0;
    
    for (const auto& feature : gallery_features_) {
        std::vector<float> normalized_feature = normalize_feature(feature);
        
        // 计算点积
        float dot_product = inner_product(
            normalized_current.begin(), normalized_current.end(),
            normalized_feature.begin(), 0.0f);
        
        // 计算模长
        float norm_current = sqrt(inner_product(
            normalized_current.begin(), normalized_current.end(),
            normalized_current.begin(), 0.0f));
        
        float norm_feature = sqrt(inner_product(
            normalized_feature.begin(), normalized_feature.end(),
            normalized_feature.begin(), 0.0f));
        
        // 计算余弦相似度
        float similarity = dot_product / (norm_current * norm_feature + 1e-8f);

        if(similarity > max_sililarity){
            max_sililarity = similarity;
        }
    }
    
    // 返回最大相似度
    return max_sililarity;
}

/**
 * @brief 重置特征图库
 * @param target_id 目标ID
 */
void GalleryManager::reset_gallery(const std::string& target_id) 
{
    gallery_features_.clear();

    // // 删除旧的图库目录
    // std::string target_gallery_dir = gallery_dir_ + "/target_" + target_id;
    // if (std::filesystem::exists(target_gallery_dir)) {
    //     std::filesystem::remove_all(target_gallery_dir);
    // }

    // int ret = system("sync;sync;sync");
    // if(ret != 0){
    //     if(m_iIsShow >= 0){
    //         std::cerr << "system(\"sync\") execute error!" << std::endl;
    //     }
    // }
    
    // // 创建新的图库目录
    // create_directory_if_not_exists(target_gallery_dir);
    // if(m_iIsShow >= 1){
    //     std::cout << "重置人员特征图库: " << target_gallery_dir << std::endl;
    // }
        
    // ret = system("sync;sync;sync");
    // if(ret != 0){
    //     if(m_iIsShow >= 0){
    //         std::cerr << "syste(\"sync\") execute error!" << std::endl;
    //     }
    // }

    return ;
}

/**
 * @brief 检查图库是否已加载
 * @return 已加载返回true，否则返回false
 */
bool GalleryManager::is_gallery_loaded() const {
    return gallery_loaded_;
}

/**
 * @brief 获取图库目标列表
 * @return 图库目标列表
 */
std::vector<GalleryManager::GalleryTarget> GalleryManager::get_gallery_targets() const {
    return gallery_targets_;
}

/**
 * @brief 获取当前加载的目标ID
 * @return 目标ID
 */
std::string GalleryManager::get_loaded_target_id() const {
    return loaded_target_id_;
}

/**
 * @brief 确保特征向量兼容性
 * @param feature 输入特征向量
 * @return 兼容的特征向量
 */
std::vector<float> GalleryManager::ensure_feature_compatibility(const std::vector<float>& feature) const {
    if (feature.empty()) {
        return {};
    }
    
    // 如果特征维度不匹配，尝试调整
    if (feature.size() != m_iFeatureVectorLen) {
        if(m_iIsShow >= 0){
            std::cerr << "警告: 特征向量维度不匹配 (" << feature.size() << " != " << m_iFeatureVectorLen << ")" << std::endl;
        }

        // 简单处理：截断或填充
        std::vector<float> compatible_feature(m_iFeatureVectorLen, 0.0f);
        size_t copy_size = std::min(feature.size(), compatible_feature.size());
        copy(feature.begin(), feature.begin() + copy_size, compatible_feature.begin());
        return compatible_feature;
    }
    
    return feature;
}

/**
 * @brief 标准化特征向量
 * @param feature 输入特征向量
 * @return 标准化后的特征向量
 */
std::vector<float> GalleryManager::normalize_feature(const std::vector<float>& feature) const 
{
    if (feature.empty()) {
        return {};
    }
    
    // 计算L2范数
    float norm = sqrt(inner_product(feature.begin(), feature.end(), feature.begin(), 0.0f));
    
    if (norm > 0.0f) {
        std::vector<float> normalized(feature.size());
        transform(feature.begin(), feature.end(), normalized.begin(),
                 [norm](float x) { return x / norm; });
        return normalized;
    }
    
    return feature;
}

/**
 * @brief 创建目录（如果不存在）
 * @param path 目录路径
 */
void GalleryManager::create_directory_if_not_exists(const std::string& path) const {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
    return ;
}
