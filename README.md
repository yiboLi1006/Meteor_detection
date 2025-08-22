# Meteor_detection
一个基于深度学习的流星轨迹识别,在风光摄影场景下,筛选出大量星空照片中含有流星轨迹的部分
# 使用方法
在./Camera_jpg_data下存放需要筛选的图像(jpg格式),直接运行./Detection.py即开始处理,结果输出于./result

由于Github无法上传较大文件,故模型参数已被拆分为多个小文件,位于./model_parts下,首次使用,需要运行./merge_model.py以合并为需要的参数文件,届时将出现./model.pth
