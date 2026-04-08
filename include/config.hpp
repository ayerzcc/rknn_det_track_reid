/**
 * @file config.hpp
 * @brief 系统配置文件 - 包含所有全局配置参数和常量定义
 * @details 支持RK3588平台优化和1024维特征向量
 */

#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <iostream>
#include <string>

#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

/**
 * @class Config
 * @brief 系统配置类，包含所有全局配置参数
 * @note 所有配置参数均为静态常量，确保线程安全
 */
class Config 
{
private:
    cv::FileStorage m_configFileHandle;
    std::string m_config_path = "";
    
    // 使用静态map存储多个配置实例
    inline static std::unordered_map<std::string, std::shared_ptr<Config>> instances;
    inline static std::mutex instances_mutex;

    Config(const std::string& configFilePath)
    {
        m_config_path = configFilePath;
        
        // 检查文件是否存在
        std::ifstream testFile(configFilePath);
        if (!testFile.good()) {
            std::cerr << "错误: 配置文件不存在或无法访问: " << configFilePath << std::endl;
            return;
        }
        testFile.close();

        m_configFileHandle = cv::FileStorage(configFilePath, cv::FileStorage::READ);
        if (!m_configFileHandle.isOpened()) 
        {
            std::cerr << "error: 无法打开YAML文件:" << m_config_path << std::endl;
            return;
        }
        
        std::cout << "成功加载配置文件: " << configFilePath << std::endl;
    }

public:
    // 删除拷贝构造和赋值
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    // 获取指定配置文件的单例
    static std::shared_ptr<Config> getInstance(const std::string& configFilePath = "./config/parameters.yaml")
    {
        std::lock_guard<std::mutex> lock(instances_mutex);
        
        auto it = instances.find(configFilePath);
        if (it != instances.end()) {
            return it->second;
        }
        
        // 创建新实例
        auto instance = std::shared_ptr<Config>(new Config(configFilePath));
        instances[configFilePath] = instance;
        return instance;
    }

    // 获取默认配置实例（向后兼容）
    static std::shared_ptr<Config> getDefaultInstance()
    {
        return getInstance("./config/parameters.yaml");
    }

    template<typename T>
    T getKeyValue(const std::string &oneStageKeyStr, const std::string &twoStageKeyStr = "") 
    {
        T value;

        if(!m_configFileHandle.isOpened()){
            std::cerr << "error: 无法打开YAML文件:" << m_config_path << std::endl;
            std::cerr << "read config file value fatal error!" << std::endl;
            return value;
        }

        if(oneStageKeyStr == "" && twoStageKeyStr == ""){
            std::cerr << "get none value for none string!" << std::endl;     
        }else if(oneStageKeyStr != "" && twoStageKeyStr == "" ){
            m_configFileHandle[oneStageKeyStr] >> value;
        }else if(oneStageKeyStr != "" && twoStageKeyStr != ""){
            m_configFileHandle[oneStageKeyStr][twoStageKeyStr] >> value;
        }else if(oneStageKeyStr == "" && twoStageKeyStr != ""){
            std::cerr << "get none value for error stage str!" << std::endl;    
        }
        
        return value;
    }
    
    // 获取当前配置文件的路径
    std::string getConfigPath() const {
        return m_config_path;
    }
    
    // 重新加载配置文件
    bool reload() {
        if (m_configFileHandle.isOpened()) {
            m_configFileHandle.release();
        }
        
        m_configFileHandle = cv::FileStorage(m_config_path, cv::FileStorage::READ);
        return m_configFileHandle.isOpened();
    }
};

// // 静态成员定义
// std::unordered_map<std::string, std::shared_ptr<Config>> Config::instances;
// std::mutex Config::instances_mutex;

inline std::string readFile2String(const std::string& filePath) 
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
};

#endif // __CONFIG_HPP__