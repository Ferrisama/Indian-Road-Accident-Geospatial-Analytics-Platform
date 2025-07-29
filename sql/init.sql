-- sql/init.sql
-- Initialize AccidentIQ database with PostGIS extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Create accidents table
CREATE TABLE IF NOT EXISTS accidents (
    accident_id VARCHAR(20) PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    latitude DECIMAL(10, 6) NOT NULL,
    longitude DECIMAL(10, 6) NOT NULL,
    geom GEOMETRY(POINT, 4326),
    accident_datetime TIMESTAMP NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL,
    hour INTEGER CHECK (hour >= 0 AND hour < 24),
    day_of_week INTEGER CHECK (day_of_week >= 0 AND day_of_week < 7),
    month INTEGER CHECK (month >= 1 AND month <= 12),
    year INTEGER CHECK (year >= 2000 AND year <= 2030),
    severity INTEGER CHECK (severity IN (1, 2, 3)),
    severity_label VARCHAR(20),
    vehicle_type VARCHAR(50),
    road_type VARCHAR(50),
    weather VARCHAR(30),
    casualties INTEGER DEFAULT 0,
    fatalities INTEGER DEFAULT 0,
    injured INTEGER DEFAULT 0,
    h3_index VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create trigger to automatically set geometry from lat/lon
CREATE OR REPLACE FUNCTION update_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER accidents_geom_trigger
    BEFORE INSERT OR UPDATE ON accidents
    FOR EACH ROW
    EXECUTE FUNCTION update_geom();

-- Create spatial index
CREATE INDEX IF NOT EXISTS idx_accidents_geom ON accidents USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_accidents_datetime ON accidents(accident_datetime);
CREATE INDEX IF NOT EXISTS idx_accidents_city ON accidents(city);
CREATE INDEX IF NOT EXISTS idx_accidents_severity ON accidents(severity);
CREATE INDEX IF NOT EXISTS idx_accidents_h3 ON accidents(h3_index);

-- Create roads table for network analysis
CREATE TABLE IF NOT EXISTS roads (
    road_id SERIAL PRIMARY KEY,
    osmid BIGINT,
    highway VARCHAR(50),
    name VARCHAR(200),
    maxspeed INTEGER,
    lanes INTEGER,
    oneway BOOLEAN DEFAULT FALSE,
    geometry GEOMETRY(LINESTRING, 4326),
    city VARCHAR(100),
    length_km DECIMAL(10, 3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_roads_geom ON roads USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_roads_highway ON roads(highway);
CREATE INDEX IF NOT EXISTS idx_roads_city ON roads(city);

-- Create accident hotspots materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS accident_hotspots AS
SELECT 
    h3_index,
    city,
    state,
    COUNT(*) as accident_count,
    AVG(severity) as avg_severity,
    SUM(casualties) as total_casualties,
    SUM(fatalities) as total_fatalities,
    SUM(injured) as total_injured,
    ST_Centroid(ST_Collect(geom)) as centroid_geom,
    EXTRACT(YEAR FROM MIN(accident_datetime)) as earliest_year,
    EXTRACT(YEAR FROM MAX(accident_datetime)) as latest_year
FROM accidents 
WHERE h3_index IS NOT NULL
GROUP BY h3_index, city, state
HAVING COUNT(*) >= 3;  -- Only hotspots with 3+ accidents

CREATE INDEX IF NOT EXISTS idx_hotspots_geom ON accident_hotspots USING GIST(centroid_geom);
CREATE INDEX IF NOT EXISTS idx_hotspots_count ON accident_hotspots(accident_count DESC);

-- Create temporal analysis view
CREATE VIEW temporal_patterns AS
SELECT 
    city,
    hour,
    day_of_week,
    month,
    COUNT(*) as accident_count,
    AVG(severity) as avg_severity,
    CASE 
        WHEN day_of_week IN (5, 6) THEN 'Weekend'
        ELSE 'Weekday'
    END as day_type,
    CASE 
        WHEN hour BETWEEN 6 AND 10 THEN 'Morning Rush'
        WHEN hour BETWEEN 17 AND 21 THEN 'Evening Rush' 
        WHEN hour BETWEEN 22 AND 5 THEN 'Night'
        ELSE 'Regular Hours'
    END as time_category
FROM accidents
GROUP BY city, hour, day_of_week, month;

-- Create severity analysis view  
CREATE VIEW severity_analysis AS
SELECT 
    city,
    state,
    vehicle_type,
    road_type,
    weather,
    COUNT(*) as total_accidents,
    SUM(CASE WHEN severity = 1 THEN 1 ELSE 0 END) as minor_accidents,
    SUM(CASE WHEN severity = 2 THEN 1 ELSE 0 END) as serious_accidents, 
    SUM(CASE WHEN severity = 3 THEN 1 ELSE 0 END) as fatal_accidents,
    ROUND(AVG(severity), 2) as avg_severity,
    SUM(casualties) as total_casualties,
    SUM(fatalities) as total_fatalities
FROM accidents
GROUP BY city, state, vehicle_type, road_type, weather;

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_accident_analytics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW accident_hotspots;
    RAISE NOTICE 'Accident analytics refreshed at %', now();
END;
$$ LANGUAGE plpgsql;

-- Create function for spatial analysis
CREATE OR REPLACE FUNCTION accidents_within_radius(
    center_lat DECIMAL,
    center_lon DECIMAL, 
    radius_meters INTEGER DEFAULT 1000
)
RETURNS TABLE(
    accident_id VARCHAR,
    city VARCHAR,
    severity INTEGER,
    distance_meters DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.accident_id,
        a.city,
        a.severity,
        ROUND(ST_Distance(
            ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326)::geography,
            a.geom::geography
        ), 2) as distance_meters
    FROM accidents a
    WHERE ST_DWithin(
        ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326)::geography,
        a.geom::geography,
        radius_meters
    )
    ORDER BY distance_meters;
END;
$$ LANGUAGE plpgsql;

-- Sample queries for testing
-- SELECT city, COUNT(*) FROM accidents GROUP BY city ORDER BY COUNT(*) DESC;
-- SELECT * FROM accidents_within_radius(19.0760, 72.8777, 5000);
-- SELECT * FROM accident_hotspots ORDER BY accident_count DESC LIMIT 10;