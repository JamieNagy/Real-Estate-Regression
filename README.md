
# Real Estate Price Prediction with Stacked Ensemble Model

This repository contains a comprehensive analysis and machine learning project aimed at predicting real estate prices using a custom stacked ensemble regression model. The project includes rigorous Exploratory Data Analysis, advanced visualizations, individual models, and a custom stacking logic designed to optimize prediction accuracy.

## Model Performance
### Individual Models
![Individual](individual%20models.png)
### Stacked Models
![Stacked](stacked.png)

## Repository Contents

- **`Real Estate Stack.ipynb`**: The main Jupyter Notebook that contains the entire analysis, from data exploration to model evaluation. The notebook includes:
  - Rigorous Exploratory Data Analysis with detailed and advanced visualizations
  - Feature Engineering and Data Preprocessing based on MI scores and visualizations from EDA.
  - Model training using five different algorithms:
    - **Random Forest Regressor**
    - **XGBoost Regressor**
    - **LightGBM Regressor**
    - **CatBoost Regressor**
    - **Artificial Neural Network**

  - **Custom stacked ensemble model** that combines the predictions from the above models for improved performance, with a focus on reducing both bias and variance.
  - **Customized Optuna study** for each model's Bayesian optimization objective, tailored to maximize the model's performance.
  - Evaluation metrics including **Adjusted R²** and **RMSLE**, with detailed analysis of model performance.

- **`train.csv`**: The training dataset contains various features of the properties along with the target variable `SalePrice`.

- **`test.csv`**: The test dataset without the target variable `SalePrice`. This is used for final predictions using the trained model.

- **`requirements.txt`**: A file listing all the Python libraries and dependencies required to run the Jupyter Notebook. 

## Data Description

The dataset includes various features related to real estate properties. Below is a brief description of the key data fields:

- `SalePrice`: The property's sale price in dollars (target variable).
- `MSSubClass`: The building class.
- `MSZoning`: The general zoning classification.
- `LotFrontage`: Linear feet of street connected to property.
- `LotArea`: Lot size in square feet.
- `Street`: Type of road access.
- `Alley`: Type of alley access.
- `LotShape`: General shape of property.
- `LandContour`: Flatness of the property.
- `Utilities`: Type of utilities available.
- `LotConfig`: Lot configuration.
- `LandSlope`: Slope of property.
- `Neighborhood`: Physical locations within Ames city limits.
- `Condition1`: Proximity to main road or railroad.
- `Condition2`: Proximity to main road or railroad (if a second is present).
- `BldgType`: Type of dwelling.
- `HouseStyle`: Style of dwelling.
- `OverallQual`: Overall material and finish quality.
- `OverallCond`: Overall condition rating.
- `YearBuilt`: Original construction date.
- `YearRemodAdd`: Remodel date.
- `RoofStyle`: Type of roof.
- `RoofMatl`: Roof material.
- `Exterior1st`: Exterior covering on house.
- `Exterior2nd`: Exterior covering on house (if more than one material).
- `MasVnrType`: Masonry veneer type.
- `MasVnrArea`: Masonry veneer area in square feet.
- `ExterQual`: Exterior material quality.
- `ExterCond`: Present condition of the material on the exterior.
- `Foundation`: Type of foundation.
- `BsmtQual`: Height of the basement.
- `BsmtCond`: General condition of the basement.
- `BsmtExposure`: Walkout or garden level basement walls.
- `BsmtFinType1`: Quality of basement finished area.
- `BsmtFinSF1`: Type 1 finished square feet.
- `BsmtFinType2`: Quality of second finished area (if present).
- `BsmtFinSF2`: Type 2 finished square feet.
- `BsmtUnfSF`: Unfinished square feet of basement area.
- `TotalBsmtSF`: Total square feet of basement area.
- `Heating`: Type of heating.
- `HeatingQC`: Heating quality and condition.
- `CentralAir`: Central air conditioning.
- `Electrical`: Electrical system.
- `1stFlrSF`: First Floor square feet.
- `2ndFlrSF`: Second floor square feet.
- `LowQualFinSF`: Low quality finished square feet (all floors).
- `GrLivArea`: Above grade (ground) living area square feet.
- `BsmtFullBath`: Basement full bathrooms.
- `BsmtHalfBath`: Basement half bathrooms.
- `FullBath`: Full bathrooms above grade.
- `HalfBath`: Half baths above grade.
- `Bedroom`: Number of bedrooms above basement level.
- `Kitchen`: Number of kitchens.
- `KitchenQual`: Kitchen quality.
- `TotRmsAbvGrd`: Total rooms above grade (does not include bathrooms).
- `Functional`: Home functionality rating.
- `Fireplaces`: Number of fireplaces.
- `FireplaceQu`: Fireplace quality.
- `GarageType`: Garage location.
- `GarageYrBlt`: Year garage was built.
- `GarageFinish`: Interior finish of the garage.
- `GarageCars`: Size of garage in car capacity.
- `GarageArea`: Size of garage in square feet.
- `GarageQual`: Garage quality.
- `GarageCond`: Garage condition.
- `PavedDrive`: Paved driveway.
- `WoodDeckSF`: Wood deck area in square feet.
- `OpenPorchSF`: Open porch area in square feet.
- `EnclosedPorch`: Enclosed porch area in square feet.
- `3SsnPorch`: Three season porch area in square feet.
- `ScreenPorch`: Screen porch area in square feet.
- `PoolArea`: Pool area in square feet.
- `PoolQC`: Pool quality.
- `Fence`: Fence quality.
- `MiscFeature`: Miscellaneous feature not covered in other categories.
- `MiscVal`: Value of miscellaneous feature.
- `MoSold`: Month Sold.
- `YrSold`: Year Sold.
- `SaleType`: Type of sale.
- `SaleCondition`: Condition of sale.


## Highlights

### Custom Stacked Ensemble Model
- **Five Base Models**: 
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - CatBoost Regressor
  - Artificial Neural Network (with TensorFlow)
  
- **Custom Stacking Logic**: Implemented a custom stacking mechanism to combine the predictions from the above models. This approach helps to reduce the model's bias and variance, leading to better generalization of unseen data.
- **Customized Loss Function for ANN**: Designed a unique loss function for the ANN to enhance performance on real estate data, effectively handling outliers and skewed distributions.
- **Customized Scoring Function for Optuna Study**: Created a custom scoring function in Optuna to optimize model hyperparameters, balancing key metrics like RMSE and RMSLE.

### Bayesian Optimization with Optuna
- Each model’s hyperparameters were fine-tuned using a **customized Optuna study** with a Bayesian optimization objective tailored to each model’s specific characteristics. This ensures that the models are optimized for the best performance.

### Performance Metrics
- **MSE**
- **RMSE**
- **RMSLE**
- **R² score** and **Adjusted R² score**

## How to Use

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JamieNagy/Real-Estate-Regression.git
   cd real-estate-regression
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:

   Open `Real Estate Stack.ipynb` in Jupyter Notebook or JupyterLab to explore the analysis, train models, and generate predictions.

4. **Predict on Test Data**:

   You can use the trained models to predict prices for the properties in `test.csv` by running the final cells in the notebook.

## Results and Performance

- **Model Performance**: Evaluated using various metrics such as **RMSLE**, RMSE, and R² score. Detailed results, including visualizations and comparisons of different models, are provided in the notebook.
- **Final Accuracy**: The final stacked model achieved an RSMLE as low as 0.11 and, an R² score and an Adjusted R² both around 0.93 with a difference of less than 0.002! This means my feature engineering is terrific.
