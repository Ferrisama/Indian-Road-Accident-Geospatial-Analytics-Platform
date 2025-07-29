#!/usr/bin/env python3
"""
AccidentIQ Data Collector
Collects road accident data from multiple free sources in India
"""

import pandas as pd
import requests
import os
from pathlib import Path
import json
import time
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccidentDataCollector:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Free data sources
        self.data_sources = {
            'government_ogd': {
                'base_url': 'https://www.data.gov.in',
                'datasets': [
                    'road-accidents-india-2020',
                    'road-accidents-india-2019',
                    'road-accidents-india-2018'
                ]
            },
            'opencity': {
                'base_url': 'https://data.opencity.in',
                'api_endpoint': '/api/3/action/datastore_search'
            }
        }

    def download_government_data(self) -> Dict:
        """Download data from data.gov.in"""
        logger.info("Downloading government accident data...")

        # Sample government data URLs (you'll need to get actual download links)
        # These are the actual datasets available on data.gov.in
        gov_datasets = {
            'accidents_2020': 'https://www.data.gov.in/sites/default/files/Road_Accidents_India_2020.csv',
            'accidents_2019': 'https://www.data.gov.in/sites/default/files/Road_Accidents_India_2019.csv',
            'state_wise_2020': 'https://www.data.gov.in/sites/default/files/State_wise_accidents_2020.csv'
        }

        downloaded_files = {}

        for dataset_name, url in gov_datasets.items():
            try:
                logger.info(f"Downloading {dataset_name}...")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    file_path = self.data_dir / f"{dataset_name}.csv"
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    downloaded_files[dataset_name] = str(file_path)
                    logger.info(f"✓ Downloaded {dataset_name}")
                else:
                    logger.warning(
                        f"Failed to download {dataset_name}: Status {response.status_code}")

            except Exception as e:
                logger.error(f"Error downloading {dataset_name}: {e}")

            time.sleep(1)  # Be respectful to the server

        return downloaded_files

    def create_sample_accident_data(self) -> str:
        """Create sample accident data for development/testing"""
        logger.info("Creating sample accident data...")

        import numpy as np
        from datetime import datetime, timedelta

        # Indian cities with approximate coordinates
        cities = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Surat': (21.1702, 72.8311),
            'Jaipur': (26.9124, 75.7873)
        }

        # Generate synthetic but realistic accident data
        np.random.seed(42)  # For reproducibility
        n_records = 10000

        data = []
        start_date = datetime(2020, 1, 1)

        for i in range(n_records):
            city = np.random.choice(list(cities.keys()))
            base_lat, base_lon = cities[city]

            # Add random offset within city boundaries (approximately ±0.1 degrees)
            lat = base_lat + np.random.uniform(-0.1, 0.1)
            lon = base_lon + np.random.uniform(-0.1, 0.1)

            # Random date within last 4 years
            days_offset = np.random.randint(0, 1460)  # 4 years
            accident_date = start_date + timedelta(days=days_offset)

            # Time patterns (more accidents during rush hours)
            hour_weights = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 1.8,
                            1.5, 1.3, 1.2, 1.1, 1.0, 1.2, 1.5, 2.2, 2.5, 2.0,
                            1.8, 1.5, 1.0, 0.7]
            hour = np.random.choice(range(24), p=np.array(
                hour_weights)/sum(hour_weights))

            accident_datetime = accident_date.replace(
                hour=hour,
                minute=np.random.randint(0, 60),
                second=np.random.randint(0, 60)
            )

            # Accident severity (1=Minor, 2=Serious, 3=Fatal)
            severity = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])

            # Vehicle types
            vehicle_types = ['Car', 'Motorcycle', 'Bus',
                             'Truck', 'Auto-rickshaw', 'Bicycle']
            vehicle_type = np.random.choice(
                vehicle_types, p=[0.4, 0.3, 0.1, 0.1, 0.08, 0.02])

            # Road types
            road_types = ['Highway', 'City Road', 'Rural Road', 'Residential']
            road_type = np.random.choice(road_types, p=[0.3, 0.4, 0.2, 0.1])

            # Weather conditions
            weather_conditions = ['Clear', 'Rain', 'Fog', 'Night']
            weather = np.random.choice(
                weather_conditions, p=[0.6, 0.2, 0.1, 0.1])

            data.append({
                'accident_id': f"ACC_{i+1:06d}",
                'city': city,
                'state': self.get_state_for_city(city),
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'accident_datetime': accident_datetime,
                'date': accident_date.date(),
                'time': accident_datetime.time(),
                'hour': hour,
                'day_of_week': accident_date.weekday(),
                'month': accident_date.month,
                'year': accident_date.year,
                'severity': severity,
                'severity_label': ['', 'Minor', 'Serious', 'Fatal'][severity],
                'vehicle_type': vehicle_type,
                'road_type': road_type,
                'weather': weather,
                # More casualties for severe accidents
                'casualties': np.random.poisson(severity),
                'fatalities': 1 if severity == 3 else (1 if np.random.random() < 0.1 else 0),
                'injured': max(0, np.random.poisson(severity) - (1 if severity == 3 else 0))
            })

        # Create DataFrame and save
        df = pd.DataFrame(data)
        sample_file = self.data_dir / "sample_accidents_india.csv"
        df.to_csv(sample_file, index=False)

        logger.info(
            f"✓ Created sample data with {len(df)} records: {sample_file}")
        return str(sample_file)

    def get_state_for_city(self, city: str) -> str:
        """Map cities to states"""
        city_state_map = {
            'Mumbai': 'Maharashtra',
            'Delhi': 'Delhi',
            'Bangalore': 'Karnataka',
            'Chennai': 'Tamil Nadu',
            'Kolkata': 'West Bengal',
            'Hyderabad': 'Telangana',
            'Pune': 'Maharashtra',
            'Ahmedabad': 'Gujarat',
            'Surat': 'Gujarat',
            'Jaipur': 'Rajasthan'
        }
        return city_state_map.get(city, 'Unknown')

    def download_osm_road_network(self, city: str = "Mumbai") -> str:
        """Download road network data from OpenStreetMap"""
        logger.info(f"Downloading road network for {city}...")

        try:
            import osmnx as ox

            # Configure osmnx
            ox.config(use_cache=True, log_console=True)

            # Download road network
            G = ox.graph_from_place(f"{city}, India", network_type='drive')

            # Convert to GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)

            # Save road network
            roads_file = self.data_dir / f"roads_{city.lower()}.geojson"
            edges.to_file(roads_file, driver='GeoJSON')

            logger.info(f"✓ Downloaded road network for {city}: {roads_file}")
            return str(roads_file)

        except Exception as e:
            logger.error(f"Error downloading road network: {e}")
            return None

    def collect_all_data(self) -> Dict:
        """Main method to collect all available data"""
        logger.info("Starting data collection...")

        collected_data = {
            'timestamp': pd.Timestamp.now(),
            'files': {}
        }

        # 1. Try to download government data
        # gov_files = self.download_government_data()
        # collected_data['files'].update(gov_files)

        # 2. Create sample data for immediate development
        sample_file = self.create_sample_accident_data()
        collected_data['files']['sample_accidents'] = sample_file

        # 3. Download road network for major cities
        major_cities = ['Mumbai', 'Delhi', 'Bangalore']
        for city in major_cities:
            road_file = self.download_osm_road_network(city)
            if road_file:
                collected_data['files'][f'roads_{city.lower()}'] = road_file

        # Save collection summary
        summary_file = self.data_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': str(collected_data['timestamp']),
                'files': collected_data['files']
            }, f, indent=2)

        logger.info(f"✓ Data collection complete. Summary: {summary_file}")
        return collected_data


if __name__ == "__main__":
    collector = AccidentDataCollector()
    results = collector.collect_all_data()

    print("\n" + "="*50)
    print("DATA COLLECTION COMPLETE")
    print("="*50)
    print(f"Files collected: {len(results['files'])}")
    for name, path in results['files'].items():
        print(f"  - {name}: {path}")
    print("\nNext steps:")
    print("1. Run: docker-compose up -d  (start database)")
    print("2. Run: python src/data_processing/load_data.py")
    print("3. Start exploring with: jupyter notebook")
