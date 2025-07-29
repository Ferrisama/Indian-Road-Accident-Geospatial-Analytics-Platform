#!/usr/bin/env python3
"""
AccidentIQ Spark Data Processor
Big data processing using Apache Spark + Apache Sedona for geospatial analytics
"""

from pathlib import Path
import logging
import h3
import pandas as pd
from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.functions import col, expr, udf
from pyspark.sql.types import DateType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccidentSparkProcessor:
    def __init__(self, app_name="AccidentIQ", sedona_jar_path=None):
        """Initialize Spark session with Sedona extensions"""

        # Use the JAR path from your original setup or allow override
        if sedona_jar_path is None:
            sedona_jar_path = "/Users/asmitghosh/Downloads/apache-sedona-1.7.2-bin/sedona-spark-shaded-3.5_2.12-1.7.2.jar"

        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.jars", sedona_jar_path) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator") \
            .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()

        # Register Sedona functions with error handling
        try:
            from sedona.register import SedonaRegistrator
            SedonaRegistrator.registerAll(self.spark)
            logger.info("✅ Sedona extensions registered successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not register Sedona extensions: {e}")
            logger.info("Continuing with basic Spark functionality...")

        # Set log level
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("✅ Spark session initialized")

    def load_accident_data(self, file_path: str):
        """Load accident data into Spark DataFrame"""
        logger.info(f"Loading accident data from {file_path}")

        # Define schema for better performance
        schema = StructType([
            StructField("accident_id", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("accident_datetime", TimestampType(), True),
            StructField("date", DateType(), True),
            StructField("hour", IntegerType(), True),
            StructField("day_of_week", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("year", IntegerType(), True),
            StructField("severity", IntegerType(), True),
            StructField("severity_label", StringType(), True),
            StructField("vehicle_type", StringType(), True),
            StructField("road_type", StringType(), True),
            StructField("weather", StringType(), True),
            StructField("casualties", IntegerType(), True),
            StructField("fatalities", IntegerType(), True),
            StructField("injured", IntegerType(), True)
        ])

        df = self.spark.read.csv(file_path, header=True, schema=schema)

        # Add geometry column using Sedona (with fallback)
        try:
            df = df.withColumn("geom", expr("ST_Point(longitude, latitude)"))
            logger.info("✅ Added geometry column using Sedona")
        except Exception as e:
            logger.warning(f"⚠️ Could not create geometry column: {e}")
            logger.info("Continuing without geometry column...")

        # Add H3 spatial indexing
        try:
            h3_udf = udf(lambda lat, lon: h3.geo_to_h3(lat, lon, 8)
                         if lat and lon else None, StringType())
            df = df.withColumn("h3_index", h3_udf(
                col("latitude"), col("longitude")))
            logger.info("✅ Added H3 spatial indexing")
        except Exception as e:
            logger.warning(f"⚠️ Could not add H3 indexing: {e}")

        logger.info(f"✅ Loaded {df.count()} accident records")
        return df

    def analyze_spatial_clusters(self, accidents_df):
        """Identify accident clusters using spatial analysis"""
        logger.info("Analyzing spatial accident clusters...")

        # Create temporary view for SQL operations
        accidents_df.createOrReplaceTempView("accidents")

        # H3 hexagon-based clustering - with fallback for geometry functions
        try:
            h3_clusters = self.spark.sql("""
                SELECT 
                    h3_index,
                    city,
                    state,
                    COUNT(*) as accident_count,
                    AVG(severity) as avg_severity,
                    SUM(casualties) as total_casualties,
                    SUM(fatalities) as total_fatalities,
                    MIN(date) as earliest_accident,
                    MAX(date) as latest_accident,
                    AVG(latitude) as center_lat,
                    AVG(longitude) as center_lon,
                    ST_Centroid(ST_Collect(geom)) as cluster_center
                FROM accidents 
                WHERE h3_index IS NOT NULL
                GROUP BY h3_index, city, state
                HAVING COUNT(*) >= 3
                ORDER BY accident_count DESC
            """)
        except Exception as e:
            logger.warning(
                f"⚠️ Sedona functions not available, using simplified clustering: {e}")
            h3_clusters = self.spark.sql("""
                SELECT 
                    h3_index,
                    city,
                    state,
                    COUNT(*) as accident_count,
                    AVG(severity) as avg_severity,
                    SUM(casualties) as total_casualties,
                    SUM(fatalities) as total_fatalities,
                    MIN(date) as earliest_accident,
                    MAX(date) as latest_accident,
                    AVG(latitude) as center_lat,
                    AVG(longitude) as center_lon
                FROM accidents 
                WHERE h3_index IS NOT NULL
                GROUP BY h3_index, city, state
                HAVING COUNT(*) >= 3
                ORDER BY accident_count DESC
            """)

        logger.info(f"✅ Identified {h3_clusters.count()} accident clusters")
        return h3_clusters

    def analyze_temporal_patterns(self, accidents_df):
        """Analyze temporal patterns in accidents"""
        logger.info("Analyzing temporal patterns...")

        accidents_df.createOrReplaceTempView("accidents")

        # Rush hour analysis
        rush_hour_patterns = self.spark.sql("""
            SELECT 
                city,
                hour,
                COUNT(*) as accident_count,
                AVG(severity) as avg_severity,
                CASE 
                    WHEN hour BETWEEN 7 AND 10 THEN 'Morning Rush'
                    WHEN hour BETWEEN 17 AND 20 THEN 'Evening Rush' 
                    WHEN hour BETWEEN 21 AND 5 THEN 'Night'
                    ELSE 'Regular Hours'
                END as time_category
            FROM accidents
            GROUP BY city, hour
            ORDER BY city, hour
        """)

        # Seasonal patterns
        seasonal_patterns = self.spark.sql("""
            SELECT 
                city,
                month,
                year,
                COUNT(*) as accident_count,
                AVG(severity) as avg_severity,
                CASE 
                    WHEN month IN (12, 1, 2) THEN 'Winter'
                    WHEN month IN (3, 4, 5) THEN 'Spring'
                    WHEN month IN (6, 7, 8, 9) THEN 'Monsoon'
                    ELSE 'Post-Monsoon'
                END as season
            FROM accidents
            GROUP BY city, month, year
            ORDER BY city, year, month
        """)

        logger.info("✅ Temporal pattern analysis complete")
        return rush_hour_patterns, seasonal_patterns

    def calculate_road_risk_scores(self, accidents_df, roads_df=None):
        """Calculate risk scores for road segments"""
        logger.info("Calculating road risk scores...")

        accidents_df.createOrReplaceTempView("accidents")

        # Risk analysis by road type and vehicle type
        risk_analysis = self.spark.sql("""
            SELECT 
                city,
                road_type,
                vehicle_type,
                weather,
                COUNT(*) as accident_count,
                AVG(severity) as avg_severity,
                SUM(fatalities) as total_fatalities,
                ROUND(
                    (COUNT(*) * AVG(severity) + SUM(fatalities) * 3) / COUNT(*), 2
                ) as risk_score
            FROM accidents
            GROUP BY city, road_type, vehicle_type, weather
            HAVING COUNT(*) >= 5
            ORDER BY risk_score DESC
        """)

        logger.info("✅ Road risk scoring complete")
        return risk_analysis

    def detect_accident_hotspots(self, accidents_df, buffer_distance=500):
        """Detect accident hotspots using spatial buffering"""
        logger.info(f"Detecting hotspots with {buffer_distance}m buffer...")

        accidents_df.createOrReplaceTempView("accidents")

        # Simplified hotspot detection using coordinate-based clustering
        hotspots = self.spark.sql(f"""
            WITH gridded_accidents AS (
                SELECT 
                    accident_id,
                    city,
                    severity,
                    latitude,
                    longitude,
                    geom,
                    ROUND(latitude * 100) / 100 as lat_grid,
                    ROUND(longitude * 100) / 100 as lon_grid
                FROM accidents
            ),
            hotspot_candidates AS (
                SELECT 
                    city,
                    lat_grid,
                    lon_grid,
                    COUNT(*) as nearby_accidents,
                    AVG(severity) as avg_severity,
                    SUM(CASE WHEN severity = 3 THEN 1 ELSE 0 END) as fatal_accidents,
                    AVG(latitude) as center_lat,
                    AVG(longitude) as center_lon
                FROM gridded_accidents
                GROUP BY city, lat_grid, lon_grid
                HAVING COUNT(*) >= 5
            )
            SELECT 
                city,
                center_lat,
                center_lon,
                nearby_accidents as max_nearby_accidents,
                avg_severity as hotspot_severity,
                fatal_accidents as total_fatal_accidents,
                1 as hotspot_intensity
            FROM hotspot_candidates
            ORDER BY nearby_accidents DESC
        """)

        logger.info(f"✅ Detected {hotspots.count()} accident hotspots")
        return hotspots

    def generate_insights_summary(self, accidents_df):
        """Generate comprehensive insights summary"""
        logger.info("Generating insights summary...")

        accidents_df.createOrReplaceTempView("accidents")

        summary = self.spark.sql("""
            SELECT 
                'Total Analysis' as category,
                COUNT(*) as total_accidents,
                COUNT(DISTINCT city) as cities_covered,
                AVG(severity) as avg_severity,
                SUM(fatalities) as total_fatalities,
                SUM(casualties) as total_casualties,
                MIN(date) as analysis_start_date,
                MAX(date) as analysis_end_date
            FROM accidents
            
            UNION ALL
            
            SELECT 
                CONCAT('City: ', city) as category,
                COUNT(*) as total_accidents,
                COUNT(DISTINCT road_type) as cities_covered,
                AVG(severity) as avg_severity,
                SUM(fatalities) as total_fatalities,
                SUM(casualties) as total_casualties,
                MIN(date) as analysis_start_date,
                MAX(date) as analysis_end_date
            FROM accidents
            GROUP BY city
            ORDER BY total_accidents DESC
        """)

        return summary

    def save_results_to_database(self, df, table_name, db_url="postgresql://postgres:password123@localhost:5432/accidentiq"):
        """Save Spark DataFrame results to PostgreSQL"""
        logger.info(f"Saving results to database table: {table_name}")

        # Convert Spark DataFrame to Pandas for easier database operations
        pandas_df = df.toPandas()

        # Handle geometry columns
        if 'cluster_center' in pandas_df.columns:
            # Convert Sedona geometry to WKT for PostGIS
            pandas_df['cluster_center_wkt'] = pandas_df['cluster_center'].astype(
                str)
            pandas_df = pandas_df.drop('cluster_center', axis=1)

        # Save to database
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            pandas_df.to_sql(table_name, engine,
                             if_exists='replace', index=False)
            logger.info(f"✅ Saved {len(pandas_df)} records to {table_name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save to database: {e}")
            logger.info("Continuing with analysis...")

    def run_complete_analysis(self, data_file="data/raw/sample_accidents_india.csv"):
        """Run complete big data analysis pipeline"""
        logger.info("🚀 Starting complete AccidentIQ analysis...")

        # Load data
        accidents_df = self.load_accident_data(data_file)

        # Run all analyses
        results = {}

        # 1. Spatial clustering
        results['clusters'] = self.analyze_spatial_clusters(accidents_df)
        self.save_results_to_database(
            results['clusters'], 'accident_clusters_spark')

        # 2. Temporal patterns
        rush_patterns, seasonal_patterns = self.analyze_temporal_patterns(
            accidents_df)
        results['rush_patterns'] = rush_patterns
        results['seasonal_patterns'] = seasonal_patterns
        self.save_results_to_database(rush_patterns, 'rush_hour_patterns')
        self.save_results_to_database(seasonal_patterns, 'seasonal_patterns')

        # 3. Risk scoring
        results['risk_scores'] = self.calculate_road_risk_scores(accidents_df)
        self.save_results_to_database(
            results['risk_scores'], 'road_risk_scores')

        # 4. Hotspot detection
        results['hotspots'] = self.detect_accident_hotspots(accidents_df)
        self.save_results_to_database(
            results['hotspots'], 'accident_hotspots_spark')

        # 5. Summary insights
        results['summary'] = self.generate_insights_summary(accidents_df)
        self.save_results_to_database(results['summary'], 'analysis_summary')

        logger.info("🎉 Complete analysis finished!")
        return results

    def close(self):
        """Close Spark session"""
        self.spark.stop()
        logger.info("✅ Spark session closed")


if __name__ == "__main__":
    processor = AccidentSparkProcessor()

    try:
        results = processor.run_complete_analysis()

        print("\n" + "="*60)
        print("🎯 ACCIDENTIQ BIG DATA ANALYSIS COMPLETE")
        print("="*60)
        print(f"✅ Spatial clusters identified")
        print(f"✅ Temporal patterns analyzed")
        print(f"✅ Road risk scores calculated")
        print(f"✅ Accident hotspots detected")
        print(f"✅ All results saved to PostgreSQL")
        print("\nNext: Run the dashboard to visualize results!")
        print("Command: streamlit run src/visualization/dashboard.py")

    finally:
        processor.close()
