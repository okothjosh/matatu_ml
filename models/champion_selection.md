# Champion Model Selection

Based on the composite score calculation, the **XGBoost** model has been selected as the champion.

## Composite Score Criteria:
The composite score was calculated using the following weights:
- 35% for RMSE (Root Mean Squared Error) - lower is better
- 35% for F1-score (macro) - higher is better
- 30% for Inference Time (ms) - lower is better

## Performance Metrics and Composite Scores:

| Model         | RMSE    | F1-score (macro) | Inference Time (ms) | Normalized_RMSE | Normalized_F1 | Normalized_Time | Composite Score |
| :------------ | :------ | :--------------- | :------------------ | :-------------- | :------------ | :-------------- | :-------------- |
| XGBoost       | 19.11 | 0.93      | 15.79         | 0.9982    | 0.9980   | 0.9911     | 0.9960   |
| LightGBM      | 19.03 | 0.93      | 55.50         | 1.0000    | 1.0000   | 0.9425     | 0.9828   |
| LSTM          | 22.51 | 0.92      | 825.94         | 0.9216    | 0.9402   | 0.0000     | 0.6516   |
| Meta_Learner  | 63.44 | 0.78      | 8.52         | 0.0000    | 0.0000   | 1.0000     | 0.3000   |

## Conclusion:
The **XGBoost** model achieved the highest composite score of 0.9960, demonstrating the best overall balance across prediction accuracy (RMSE, F1-score) and operational efficiency (inference time).
