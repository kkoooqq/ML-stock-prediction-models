# ML-stock-prediction-models

使用机器学习进行股票预测并指导短线交易，此处使用分类算法。
* 准备纯量价数据，不考虑基本面信息
* 特征工程，自己构建量价关系特征，结合资金流量，并使用talib一些K线形态做特征
* 数据处理（中位数去极值，行业中性化，标准化）
* 数据源准备，回看15日数据/预测未来3日涨跌幅进行分类(>5%, 0~5%, -5%~0, <-5%，分四类)，标记标签
* 将数据拆分为70%训练集，30%验证集
* 移除相关性大的特征![移除特征](https://user-images.githubusercontent.com/43202488/136674627-d8976909-f0ed-4cde-be52-8550aac5d941.png)
* 进行重要特征选择![重要特征选择](https://user-images.githubusercontent.com/43202488/136674673-51b2f4d2-6e9a-48bc-904f-f2d883f1051c.png)
![cumulativeFeatureImportance](https://user-images.githubusercontent.com/43202488/136674677-e1506539-b7b0-45ab-81db-a5de67f7f7f4.png)
* 显著特征比例选择![显著特征比例](https://user-images.githubusercontent.com/43202488/136674699-4f2bbf43-66c2-4686-bdea-fbf5961da8a5.png)
* 构建LGB模型
* 贝叶斯优化
* 根据最优参数训练模型
* 验证集预测结果
* 使用模型进行选股
* 对选股结果进行指标评估![指标评估](https://user-images.githubusercontent.com/43202488/136674773-51fa352c-c7c6-4891-847d-968628187897.png)
* 最终投资组合分析![最终投资组合分析](https://user-images.githubusercontent.com/43202488/136674929-28ccdb60-acc9-4522-bd08-a568f1f35fe3.png)

