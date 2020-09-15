# AI水印
利用对抗样本技术对图片进行定向生成，用于检测图片是否被盗用。可利用AI识别对抗样本来判断对方是否盗用图片。
进阶版可参考《Hidden Trigger Backdoor Attacks》中的方法，利用对抗样本进行投毒实现后门攻击。

# 实验环境
python 3.6
tensorflow 1.13.1

# 使用说明
git clone url
python AIsec001.py image1_path image2_path
image1_path:干净图片
image2_path:目标图片
