import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://postgres:password123@localhost:5432/accidentiq')

query = '''
SELECT 
    city,
    COUNT(*) AS total_accidents,
    AVG(severity) AS avg_severity,
    SUM(fatalities) AS total_fatalities
FROM accidents 
GROUP BY city 
ORDER BY total_accidents DESC 
LIMIT 5;
'''

try:
    df = pd.read_sql(query, engine)
    print('📊 Top 5 cities by accident count:')
    print(df.to_string(index=False))
    print('\n✅ Database setup successful!')
except Exception as e:
    print(f'❌ Error: {e}')
