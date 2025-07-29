#!/usr/bin/env python3
"""
AccidentIQ Machine Learning Models
Predictive models for accident severity, hotspot detection, and risk assessment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sqlalchemy import create_engine
import h3
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccidentMLPredictor:
    def __init__(self, db_url="postgresql://postgres:password123@localhost:5432/accidentiq"):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    def load_and_prepare_data(self):
        """Load and prepare data for machine learning"""
        logger.info("Loading and preparing data for ML...")

        query = """
        SELECT 
            accident_id, city, state, latitude, longitude,
            EXTRACT(HOUR FROM accident_datetime) as hour,
            EXTRACT(DOW FROM accident_datetime) as day_of_week,
            EXTRACT(MONTH FROM accident_datetime) as month,
            EXTRACT(YEAR FROM accident_datetime) as year,
            severity, vehicle_type, road_type, weather,
            casualties, fatalities, injured, h3_index
        FROM accidents
        WHERE latitude IS NOT NULL 
        AND longitude IS NOT NULL
        AND severity IS NOT NULL
        """

        df = pd.read_sql(query, self.engine)

        # Feature engineering
        df = self._create_features(df)

        logger.info(f"✅ Loaded {len(df)} records with {df.shape[1]} features")
        return df

    def _create_features(self, df):
        """Create additional features for machine learning"""
        logger.info("Creating engineered features...")

        # Time-based features
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 10)) | (
            df['hour'].between(17, 20))).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)

        # Seasonal features
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
            10: 'Post-Monsoon', 11: 'Post-Monsoon'
        })

        # H3 spatial features
        if 'h3_index' in df.columns:
            # Accident density in nearby hexagons
            h3_counts = df['h3_index'].value_counts()
            df['h3_accident_density'] = df['h3_index'].map(h3_counts).fillna(0)

            # Add neighbor hexagon density
            def get_neighbor_density(h3_idx):
                try:
                    # Get neighboring hexagons
                    neighbors = h3.k_ring(h3_idx, 1)
                    neighbor_counts = [h3_counts.get(
                        neighbor, 0) for neighbor in neighbors]
                    return sum(neighbor_counts) / len(neighbor_counts)
                except:
                    return 0

            df['neighbor_accident_density'] = df['h3_index'].apply(
                get_neighbor_density)

        # City-based features
        city_stats = df.groupby('city').agg({
            'severity': ['mean', 'std'],
            'casualties': 'mean'
        }).reset_index()
        city_stats.columns = ['city', 'city_avg_severity',
                              'city_severity_std', 'city_avg_casualties']
        df = df.merge(city_stats, on='city', how='left')

        # Road type risk scores
        road_risk = df.groupby('road_type')['severity'].mean()
        df['road_type_risk'] = df['road_type'].map(road_risk)

        # Vehicle type risk scores
        vehicle_risk = df.groupby('vehicle_type')['severity'].mean()
        df['vehicle_type_risk'] = df['vehicle_type'].map(vehicle_risk)

        logger.info(f"✅ Created {df.shape[1]} total features")
        return df

    def train_severity_prediction_model(self, df):
        """Train model to predict accident severity"""
        logger.info("Training accident severity prediction model...")

        # Prepare features
        feature_columns = [
            'latitude', 'longitude', 'hour', 'day_of_week', 'month',
            'is_weekend', 'is_rush_hour', 'is_night',
            'h3_accident_density', 'neighbor_accident_density',
            'city_avg_severity', 'city_severity_std', 'city_avg_casualties',
            'road_type_risk', 'vehicle_type_risk'
        ]

        # Encode categorical variables
        categorical_features = ['city', 'state',
                                'vehicle_type', 'road_type', 'weather', 'season']

        df_encoded = df.copy()
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df_encoded[f'{feature}_encoded'] = le.fit_transform(
                    df[feature].astype(str))
                feature_columns.append(f'{feature}_encoded')
                self.encoders[feature] = le

        # Prepare data
        X = df_encoded[feature_columns].fillna(0)
        y = df_encoded['severity'] - 1  # Convert to 0, 1, 2 for classification

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['severity_model'] = scaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train XGBoost model with hyperparameter tuning
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=42,
            eval_metric='mlogloss'
        )

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate model
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)

        logger.info(
            f"✅ Severity Model - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("Top 5 Important Features:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        # Save model
        self.models['severity_prediction'] = {
            'model': best_model,
            'features': feature_columns,
            'accuracy': test_score,
            'feature_importance': feature_importance
        }

        # Generate detailed evaluation
        y_pred = best_model.predict(X_test)
        classification_rep = classification_report(y_test, y_pred,
                                                   target_names=['Minor', 'Serious', 'Fatal'])
        logger.info(f"Classification Report:\n{classification_rep}")

        return best_model, test_score

    def train_hotspot_prediction_model(self, df):
        """Train model to predict accident hotspots"""
        logger.info("Training hotspot prediction model...")

        # Create grid-based features for hotspot prediction
        # Aggregate accidents by H3 hexagons
        h3_data = df.groupby('h3_index').agg({
            'accident_id': 'count',
            'severity': 'mean',
            'fatalities': 'sum',
            'casualties': 'sum',
            'hour': lambda x: x.mode().iloc[0],  # Most common hour
            'vehicle_type': lambda x: x.mode().iloc[0],  # Most common vehicle
            'road_type': lambda x: x.mode().iloc[0],  # Most common road type
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()

        h3_data.columns = ['h3_index', 'accident_count', 'avg_severity', 'total_fatalities',
                           'total_casualties', 'common_hour', 'common_vehicle', 'common_road',
                           'center_lat', 'center_lon']

        # Define hotspot threshold (top 20% by accident count)
        threshold = h3_data['accident_count'].quantile(0.8)
        h3_data['is_hotspot'] = (
            h3_data['accident_count'] >= threshold).astype(int)

        # Feature engineering for hexagons
        h3_data['density_score'] = h3_data['accident_count'] / \
            h3_data['accident_count'].max()
        h3_data['severity_risk'] = h3_data['avg_severity'] * \
            h3_data['total_fatalities']

        # Prepare features
        feature_columns = ['center_lat', 'center_lon',
                           'avg_severity', 'density_score', 'common_hour']

        # Encode categorical features
        for col in ['common_vehicle', 'common_road']:
            le = LabelEncoder()
            h3_data[f'{col}_encoded'] = le.fit_transform(
                h3_data[col].astype(str))
            feature_columns.append(f'{col}_encoded')
            self.encoders[f'hotspot_{col}'] = le

        X = h3_data[feature_columns].fillna(0)
        y = h3_data['is_hotspot']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['hotspot_model'] = scaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train LightGBM model
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            n_estimators=200,
            learning_rate=0.1
        )

        lgb_model.fit(X_train, y_train)

        # Evaluate
        train_score = lgb_model.score(X_train, y_train)
        test_score = lgb_model.score(X_test, y_test)

        logger.info(
            f"✅ Hotspot Model - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")

        # Save model
        self.models['hotspot_prediction'] = {
            'model': lgb_model,
            'features': feature_columns,
            'accuracy': test_score,
            'threshold': threshold
        }

        return lgb_model, test_score

    def train_casualty_prediction_model(self, df):
        """Train model to predict number of casualties"""
        logger.info("Training casualty prediction model...")

        # Prepare features (similar to severity model)
        feature_columns = [
            'latitude', 'longitude', 'hour', 'day_of_week', 'month',
            'severity', 'is_weekend', 'is_rush_hour', 'is_night',
            'h3_accident_density', 'neighbor_accident_density',
            'road_type_risk', 'vehicle_type_risk'
        ]

        # Encode categorical variables
        categorical_features = ['city', 'vehicle_type', 'road_type', 'weather']

        df_encoded = df.copy()
        for feature in categorical_features:
            if feature in df.columns:
                if f'casualty_{feature}' not in self.encoders:
                    le = LabelEncoder()
                    df_encoded[f'{feature}_encoded'] = le.fit_transform(
                        df[feature].astype(str))
                    self.encoders[f'casualty_{feature}'] = le
                else:
                    le = self.encoders[f'casualty_{feature}']
                    df_encoded[f'{feature}_encoded'] = le.transform(
                        df[feature].astype(str))
                feature_columns.append(f'{feature}_encoded')

        # Prepare data
        X = df_encoded[feature_columns].fillna(0)
        y = df_encoded['casualties']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['casualty_model'] = scaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # Evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"✅ Casualty Model - MSE: {mse:.3f}, R²: {r2:.3f}")

        # Save model
        self.models['casualty_prediction'] = {
            'model': rf_model,
            'features': feature_columns,
            'mse': mse,
            'r2_score': r2
        }

        return rf_model, r2

    def predict_accident_risk(self, lat, lon, hour, vehicle_type, road_type, weather='Clear', city='Unknown', state='Unknown'):
        """Predict accident risk for given parameters"""
        if 'severity_prediction' not in self.models:
            raise ValueError("Severity prediction model not trained!")

        # Create input data
        input_data = pd.DataFrame({
            'latitude': [lat],
            'longitude': [lon],
            'hour': [hour],
            'day_of_week': [pd.Timestamp.now().weekday()],
            'month': [pd.Timestamp.now().month],
            'city': [city],
            'state': [state],
            'vehicle_type': [vehicle_type],
            'road_type': [road_type],
            'weather': [weather]
        })

        # Add engineered features
        input_data['is_weekend'] = (
            input_data['day_of_week'].isin([5, 6])).astype(int)
        input_data['is_rush_hour'] = ((input_data['hour'].between(7, 10)) |
                                      (input_data['hour'].between(17, 20))).astype(int)
        input_data['is_night'] = ((input_data['hour'] < 6) | (
            input_data['hour'] > 22)).astype(int)

        # Add season
        input_data['season'] = input_data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
            10: 'Post-Monsoon', 11: 'Post-Monsoon'
        })

        # Add dummy values for other features (these would normally be calculated from historical data)
        input_data['h3_accident_density'] = 0
        input_data['neighbor_accident_density'] = 0
        input_data['city_avg_severity'] = 2.0  # Default average
        input_data['city_severity_std'] = 0.5   # Default std
        input_data['city_avg_casualties'] = 1.5  # Default casualties
        input_data['road_type_risk'] = 2.0      # Default risk
        input_data['vehicle_type_risk'] = 2.0   # Default risk

        # Encode ALL categorical features that were used during training
        categorical_features = ['city', 'state',
                                'vehicle_type', 'road_type', 'weather', 'season']

        for feature in categorical_features:
            if feature in self.encoders:
                try:
                    # Try to encode the value
                    encoded_value = self.encoders[feature].transform(
                        [input_data[feature].iloc[0]])[0]
                    input_data[f'{feature}_encoded'] = encoded_value
                except ValueError:
                    # If the value wasn't seen during training, use a default (0 or most common)
                    print(
                        f"Warning: Unknown {feature} value '{input_data[feature].iloc[0]}', using default encoding")
                    input_data[f'{feature}_encoded'] = 0
            else:
                # If encoder doesn't exist, use default
                input_data[f'{feature}_encoded'] = 0

        # Get features in correct order and ensure all are present
        model_features = self.models['severity_prediction']['features']

        # Create a dataframe with all required features, filling missing ones with 0
        feature_data = {}
        for feature in model_features:
            if feature in input_data.columns:
                feature_data[feature] = input_data[feature].iloc[0]
            else:
                feature_data[feature] = 0
                print(
                    f"Warning: Feature '{feature}' not found, using default value 0")

        X = pd.DataFrame([feature_data])

        # Scale and predict
        X_scaled = self.scalers['severity_model'].transform(X)
        severity_pred = self.models['severity_prediction']['model'].predict(X_scaled)[
            0]
        severity_proba = self.models['severity_prediction']['model'].predict_proba(X_scaled)[
            0]

        return {
            # Convert back to 1,2,3
            'predicted_severity': int(severity_pred + 1),
            'severity_probabilities': {
                'Minor': float(severity_proba[0]),
                'Serious': float(severity_proba[1]),
                'Fatal': float(severity_proba[2])
            },
            # Weighted risk score
            'risk_score': float(np.dot(severity_proba, [1, 2, 3]))
        }

    def save_models(self):
        """Save all trained models"""
        logger.info("Saving trained models...")

        for model_name, model_info in self.models.items():
            joblib.dump(model_info, self.model_dir / f"{model_name}.pkl")

        # Save encoders and scalers
        joblib.dump(self.encoders, self.model_dir / "encoders.pkl")
        joblib.dump(self.scalers, self.model_dir / "scalers.pkl")

        logger.info(f"✅ Models saved to {self.model_dir}")

    def load_models(self):
        """Load pre-trained models"""
        logger.info("Loading pre-trained models...")

        try:
            for model_file in self.model_dir.glob("*.pkl"):
                if model_file.stem in ['encoders', 'scalers']:
                    continue
                self.models[model_file.stem] = joblib.load(model_file)

            # Load encoders and scalers
            if (self.model_dir / "encoders.pkl").exists():
                self.encoders = joblib.load(self.model_dir / "encoders.pkl")
            if (self.model_dir / "scalers.pkl").exists():
                self.scalers = joblib.load(self.model_dir / "scalers.pkl")

            logger.info(f"✅ Loaded {len(self.models)} models")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def generate_city_risk_report(self, city_name):
        """Generate comprehensive risk report for a city"""
        logger.info(f"Generating risk report for {city_name}...")

        query = f"""
        SELECT * FROM accidents 
        WHERE city = '{city_name}'
        AND latitude IS NOT NULL 
        AND longitude IS NOT NULL
        """

        city_df = pd.read_sql(query, self.engine)

        if city_df.empty:
            return {"error": f"No data found for {city_name}"}

        # Calculate risk metrics
        report = {
            'city': city_name,
            'total_accidents': len(city_df),
            'fatal_accidents': len(city_df[city_df['severity'] == 3]),
            'avg_severity': city_df['severity'].mean(),
            'total_casualties': city_df['casualties'].sum(),
            'fatality_rate': city_df['fatalities'].sum() / len(city_df) * 100,
            'peak_accident_hour': city_df['hour'].mode().iloc[0],
            'riskiest_road_type': city_df.groupby('road_type')['severity'].mean().idxmax(),
            'riskiest_vehicle_type': city_df.groupby('vehicle_type')['severity'].mean().idxmax(),
            'seasonal_pattern': city_df.groupby('month').size().to_dict(),
            'recommendations': []
        }

        # Generate recommendations
        if report['fatality_rate'] > 5:
            report['recommendations'].append(
                "CRITICAL: High fatality rate - implement emergency safety measures")

        if report['avg_severity'] > 2.0:
            report['recommendations'].append(
                "HIGH: Install speed monitoring and traffic calming measures")

        peak_hour = report['peak_accident_hour']
        if 7 <= peak_hour <= 10 or 17 <= peak_hour <= 20:
            report['recommendations'].append(
                f"MEDIUM: Rush hour accidents peak at {peak_hour}:00 - optimize traffic signals")

        # Hotspot locations
        if 'h3_index' in city_df.columns:
            hotspots = city_df.groupby('h3_index').agg({
                'accident_id': 'count',
                'latitude': 'mean',
                'longitude': 'mean',
                'severity': 'mean'
            }).sort_values('accident_id', ascending=False).head(5)

            report['top_hotspots'] = []
            for h3_idx, data in hotspots.iterrows():
                report['top_hotspots'].append({
                    'h3_index': h3_idx,
                    'accident_count': int(data['accident_id']),
                    'latitude': float(data['latitude']),
                    'longitude': float(data['longitude']),
                    'avg_severity': float(data['severity'])
                })

        return report


def main():
    """Main function to train all models"""
    predictor = AccidentMLPredictor()

    try:
        # Load data
        df = predictor.load_and_prepare_data()

        if df.empty:
            logger.error("No data available for training!")
            return

        logger.info(f"🚀 Starting ML pipeline with {len(df)} records...")

        # Train models
        models_trained = {}

        # 1. Severity prediction
        severity_model, severity_score = predictor.train_severity_prediction_model(
            df)
        models_trained['severity'] = severity_score

        # 2. Hotspot prediction
        hotspot_model, hotspot_score = predictor.train_hotspot_prediction_model(
            df)
        models_trained['hotspot'] = hotspot_score

        # 3. Casualty prediction
        casualty_model, casualty_r2 = predictor.train_casualty_prediction_model(
            df)
        models_trained['casualty'] = casualty_r2

        # Save models
        predictor.save_models()

        # Generate sample predictions
        logger.info("🧪 Testing model predictions...")

        # Test prediction for Mumbai coordinates
        test_prediction = predictor.predict_accident_risk(
            lat=19.0760, lon=72.8777, hour=18,
            vehicle_type='Car', road_type='City Road', weather='Clear',
            city='Mumbai', state='Maharashtra'  # Add these parameters
        )

        logger.info(f"Sample prediction: {test_prediction}")

        # Generate city report
        cities = df['city'].unique()[:3]  # Test with first 3 cities
        for city in cities:
            report = predictor.generate_city_risk_report(city)
            logger.info(f"Risk report for {city}: {report['total_accidents']} accidents, "
                        f"{report['fatality_rate']:.1f}% fatality rate")

        print("\n" + "="*70)
        print("🎯 ACCIDENTIQ MACHINE LEARNING PIPELINE COMPLETE")
        print("="*70)
        print("✅ Models Trained Successfully:")
        print(
            f"   📊 Severity Prediction: {models_trained['severity']:.3f} accuracy")
        print(
            f"   🔥 Hotspot Detection: {models_trained['hotspot']:.3f} accuracy")
        print(
            f"   🚑 Casualty Prediction: {models_trained['casualty']:.3f} R² score")
        print(f"\n✅ Models saved to: models/")
        print(f"✅ Ready for real-time predictions!")
        print("\nNext steps:")
        print("1. Integrate with dashboard: Add ML predictions to Streamlit app")
        print("2. API endpoint: Create Flask/FastAPI for real-time predictions")
        print("3. Batch predictions: Schedule daily risk assessments")

    except Exception as e:
        logger.error(f"Error in ML pipeline: {e}")
        raise


if __name__ == "__main__":
    main
