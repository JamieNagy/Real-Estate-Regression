
# Real Estate Price Prediction with Stacked Ensemble Model

This repository contains a comprehensive analysis and machine learning project aimed at predicting real estate prices using a custom stacked ensemble regression model. The project includes rigorous Exploratory Data Analysis (EDA), advanced visualizations, individual models, and a custom stacking logic designed to optimize prediction accuracy.

## Repository Contents

- **`Real Estate Stack.ipynb`**: The main Jupyter Notebook that contains the entire analysis, from data exploration to model evaluation. The notebook includes:
  - Data loading and cleaning
  - **Rigorous Exploratory Data Analysis (EDA)** with detailed and advanced visualizations
  - Feature engineering and selection
  - Model training using five different algorithms:
    - **Random Forest Regressor**
    - **XGBoost Regressor**
    - **LightGBM Regressor**
    - **CatBoost Regressor**
    - **Artificial Neural Network (ANN)**
  - **Custom stacked ensemble model** that combines the predictions from the above models for improved performance, with a focus on reducing both bias and variance.
  - **Customized Optuna study** for each model's Bayesian optimization objective, tailored to maximize the model's performance.
  - Evaluation metrics including RMSE (Root Mean Squared Error) and **RMSLE (Root Mean Squared Logarithmic Error)**, with detailed analysis of model performance.

- **`train.csv`**: The training dataset containing various features of the properties along with the target variable `SalePrice`.

- **`test.csv`**: The test dataset without the target variable `SalePrice`. This is used for final predictions using the trained model.

- **`requirements.txt`**: A file listing all the Python libraries and dependencies required to run the Jupyter Notebook. 

## Data Description

The dataset includes various features related to real estate properties. Below is a brief description of the key data fields:

- **SalePrice**: The property's sale price in dollars (target variable).
- **MSSubClass**: The building class.
- **MSZoning**: The general zoning classification.
- **LotFrontage**: Linear feet of street connected to property.
- **LotArea**: Lot size in square feet.
- **Street**: Type of road access.
- **Alley**: Type of alley access.
- **LotShape**: General shape of property.
- **LandContour**: Flatness of the property.
- **Utilities**: Type of utilities available.
- **LotConfig**: Lot configuration.
- **LandSlope**: Slope of property.
- **Neighborhood**: Physical locations within Ames city limits.
- **Condition1**: Proximity to main road or railroad.
- **Condition2**: Proximity to main road or railroad (if a second is present).
- **BldgType**: Type of dwelling.
- **HouseStyle**: Style of dwelling.
- **OverallQual**: Overall material and finish quality.
- **OverallCond**: Overall condition rating.
- **YearBuilt**: Original construction date.
- **YearRemodAdd**: Remodel date.
- **RoofStyle**: Type of roof.
- **RoofMatl**: Roof material.
- **Exterior1st**: Exterior covering on house.
- **Exterior2nd**: Exterior covering on house (if more than one material).
- **MasVnrType**: Masonry veneer type.
- **MasVnrArea**: Masonry veneer area in square feet.
- **ExterQual**: Exterior material quality.
- **ExterCond**: Present condition of the material on the exterior.
- **Foundation**: Type of foundation.
- **BsmtQual**: Height of the basement.
- **BsmtCond**: General condition of the basement.
- **BsmtExposure**: Walkout or garden level basement walls.
- **BsmtFinType1**: Quality of basement finished area.
- **BsmtFinSF1**: Type 1 finished square feet.
- **BsmtFinType2**: Quality of second finished area (if present).
- **BsmtFinSF2**: Type 2 finished square feet.
- **BsmtUnfSF**: Unfinished square feet of basement area.
- **TotalBsmtSF**: Total square feet of basement area.
- **Heating**: Type of heating.
- **HeatingQC**: Heating quality and condition.
- **CentralAir**: Central air conditioning.
- **Electrical**: Electrical system.
- **1stFlrSF**: First Floor square feet.
- **2ndFlrSF**: Second floor square feet.
- **LowQualFinSF**: Low quality finished square feet (all floors).
- **GrLivArea**: Above grade (ground) living area square feet.
- **BsmtFullBath**: Basement full bathrooms.
- **BsmtHalfBath**: Basement half bathrooms.
- **FullBath**: Full bathrooms above grade.
- **HalfBath**: Half baths above grade.
- **Bedroom**: Number of bedrooms above basement level.
- **Kitchen**: Number of kitchens.
- **KitchenQual**: Kitchen quality.
- **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms).
- **Functional**: Home functionality rating.
- **Fireplaces**: Number of fireplaces.
- **FireplaceQu**: Fireplace quality.
- **GarageType**: Garage location.
- **GarageYrBlt**: Year garage was built.
- **GarageFinish**: Interior finish of the garage.
- **GarageCars**: Size of garage in car capacity.
- **GarageArea**: Size of garage in square feet.
- **GarageQual**: Garage quality.
- **GarageCond**: Garage condition.
- **PavedDrive**: Paved driveway.
- **WoodDeckSF**: Wood deck area in square feet.
- **OpenPorchSF**: Open porch area in square feet.
- **EnclosedPorch**: Enclosed porch area in square feet.
- **3SsnPorch**: Three season porch area in square feet.
- **ScreenPorch**: Screen porch area in square feet.
- **PoolArea**: Pool area in square feet.
- **PoolQC**: Pool quality.
- **Fence**: Fence quality.
- **MiscFeature**: Miscellaneous feature not covered in other categories.
- **MiscVal**: Value of miscellaneous feature.
- **MoSold**: Month Sold.
- **YrSold**: Year Sold.
- **SaleType**: Type of sale.
- **SaleCondition**: Condition of sale.

For a complete description of the data fields, please refer to the data description file in the dataset.

## Highlights

### Custom Stacked Ensemble Model
- **Five Base Models**: 
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - CatBoost Regressor
  - Artificial Neural Network (ANN)
  
- **Custom Stacking Logic**: Implemented a custom stacking mechanism to combine the predictions from the above models. This approach helps to reduce the model's bias and variance, leading to better generalization on unseen data.

### Bayesian Optimization with Optuna
- Each model’s hyperparameters were fine-tuned using a **customized Optuna study** with a Bayesian optimization objective tailored to each model’s specific characteristics. This ensures that the models are optimized for the best performance.

### Performance Metrics
- **RMSLE (Root Mean Squared Logarithmic Error)**: This metric is emphasized for its ability to handle skewed distributions in price predictions, making it a critical measure for this project.
- **RMSE (Root Mean Squared Error)** and **R² score**: Also included for comprehensive model evaluation and comparison.

## Requirements

To run the notebook and replicate the analysis, you'll need to install the required Python libraries listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/real-estate-regression.git
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

The model's performance is evaluated using various metrics such as **RMSLE**, RMSE, and R² score. Detailed results, including visualizations and comparisons of different models, are provided in the notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
