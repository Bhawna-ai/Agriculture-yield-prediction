import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def setup_dask_client():
    """Setup Dask distributed client for parallel processing"""
    print("Setting up Dask distributed client...")
    try:
        cluster = LocalCluster(n_workers=4)
        client = Client(cluster)
        print(f"Dashboard link: {client.dashboard_link}")
        return client
    except Exception as e:
        print(f"Error setting up Dask client: {e}")
        return None

def prepare_data(file_path):
    """Load and prepare data using Dask"""
    print("Loading and preprocessing data...")
    
    # Load data with Dask
    df = dd.read_csv(file_path)
    print("Data loaded successfully.")
    
    # Convert to pandas for encoding
    df = df.compute()
    
    # Handle missing values
    categorical_columns = ['State', 'District ', 'Crop', 'Season']
    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')
    
    # Create essential features only
    print("Creating essential features...")
    
    # 1. Basic area features
    df['Area_Ratio'] = df['Area '] / df.groupby('State')['Area '].transform('mean')
    
    # 2. Historical patterns
    df['Historical_Yield'] = df.groupby('Crop')['Yield'].transform('mean')
    
    # 3. Seasonal patterns
    df['Seasonal_Yield'] = df.groupby(['Crop', 'Season'])['Yield'].transform('mean')
    
    # 4. Regional patterns
    df['State_Yield_Mean'] = df.groupby('State')['Yield'].transform('mean')
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_columns:
        print(f"Encoding {col}...")
        encoders[col] = LabelEncoder()
        df[f"{col}_encoded"] = encoders[col].fit_transform(df[col])
        
        # Add mean yield encoding
        mean_yield = df.groupby(col)['Yield'].mean()
        df[f"{col}_yield_mean"] = df[col].map(mean_yield)
    
    # Select essential features
    feature_columns = [
        'Area ', 'Area_Ratio',
        'Historical_Yield', 'Seasonal_Yield',
        'State_Yield_Mean'
    ] + [f"{col}_encoded" for col in categorical_columns] + \
       [f"{col}_yield_mean" for col in categorical_columns]
    
    X = df[feature_columns]
    y = df['Yield']
    
    return X, y, encoders, feature_columns, df

def train_model(X, y):
    """Train an optimized Random Forest model with cross-validation"""
    print("Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    # Train model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=100,        # Reduced number of trees
        max_depth=15,            # Reduced max depth
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Perform quick cross-validation
    print("\nPerforming quick cross-validation...")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Check for overfitting
    train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    print(f"\nTraining R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    print(f"R² Score Difference: {abs(train_r2 - r2):.4f}")
    
    # Create prediction error plot
    plt.figure(figsize=(10, 6))
    errors = y_test - y_pred
    plt.scatter(y_test, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error Analysis')
    plt.tight_layout()
    plt.savefig('prediction_error.png')
    plt.close()
    
    return model, X_test, y_test, y_pred

def create_visualizations(model, X_test, y_test, y_pred, feature_cols, df):
    """Create and save visualization plots"""
    print("Creating visualizations...")
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # 2. Actual vs Predicted Plot with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    
    # Add confidence intervals
    errors = y_test - y_pred
    std_error = np.std(errors)
    plt.fill_between([y_test.min(), y_test.max()],
                    [y_test.min() - 2*std_error, y_test.max() - 2*std_error],
                    [y_test.min() + 2*std_error, y_test.max() + 2*std_error],
                    alpha=0.2, color='gray')
    
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs Predicted Yield with 95% Confidence Interval')
    plt.tight_layout()
    plt.savefig('prediction_performance.png')
    plt.close()
    
    # 3. Yield Distribution by Crop
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='Crop', y='Yield', data=df)
    plt.xticks(rotation=45)
    plt.title('Yield Distribution by Crop')
    plt.tight_layout()
    plt.savefig('yield_distribution.png')
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation = df[[col for col in df.columns if '_encoded' in col or col == 'Yield' or col == 'Area ']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    print("Visualizations saved as PNG files.")

def save_model(model, encoders, feature_cols):
    """Save the model and encoders"""
    print("Saving model...")
    joblib.dump(model, 'crop_yield_model.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    joblib.dump(feature_cols, 'feature_cols.joblib')
    print("Model saved successfully!")

def main():
    client = None
    try:
        # Setup Dask client
        client = setup_dask_client()
        
        # Prepare data
        X, y, encoders, feature_cols, df = prepare_data('crop_yield_train.csv')
        
        # Train model
        model, X_test, y_test, y_pred = train_model(X, y)
        
        # Create visualizations
        create_visualizations(model, X_test, y_test, y_pred, feature_cols, df)
        
        # Save model
        save_model(model, encoders, feature_cols)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if client:
            client.close()
            print("Dask client closed.")

if __name__ == "__main__":
    main() 