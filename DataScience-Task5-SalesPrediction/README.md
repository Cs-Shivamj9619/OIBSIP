# Task 5: Sales Prediction Using Python

## Project Overview
This project builds predictive regression models to forecast product sales based on advertising platform expenditures across TV, Radio, and Newspaper channels.

## Dataset
- **Source**: Advertising Spend & Sales Dataset
- **Variables**:
  - `TV`: Advertising spend on television
  - `Radio`: Advertising spend on radio
  - `Newspaper`: Advertising spend on newspapers
  - `Sales`: Total units sold (Target Variable)

## Methodology
1. **Exploratory Data Analysis**: Analyzed individual media correlation with product sales using regression plots and correlation heatmaps.
2. **Data Splitting**: 80/20 train/test split.
3. **Model Development**:
   - **Linear Regression** (Interpretable baseline model)
   - **Random Forest Regressor** (Non-linear ensemble model)
4. **Evaluation Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score.

## Model Performance
| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Linear Regression** | ~1.46 | ~1.78 | ~0.899 |
| **Random Forest Regressor** | ~0.62 | ~0.79 | ~0.981 |

## Business Insights
- **TV advertising** has the strongest linear correlation with sales revenue, followed closely by **Radio**.
- **Newspaper advertising** demonstrates the weakest correlation and coefficient impact, indicating budget reallocation toward TV and Radio channels maximizes sales returns.
