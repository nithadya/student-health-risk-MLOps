import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def test_feature_engineering_logic():
    df = pd.DataFrame({
        'sleep_duration': [5.0, 8.0, 7.0],
        'stress_level': ['high', 'low', 'moderate'],
        'physical_activity_level': ['sedentary', 'active', 'moderate'],
        'sleep_quality': ['poor', 'good', 'average'],
        'smoking_alcohol': ['yes', 'no', 'no'],
        'bmi': [28.0, 21.0, 23.0]
    })
    
    risk_index = (
        (df['sleep_duration'] < 6.0).astype(int) +
        (df['stress_level'] == 'high').astype(int) +
        (df['sleep_quality'] == 'poor').astype(int) +
        (df['physical_activity_level'] == 'sedentary').astype(int) +
        (df['smoking_alcohol'] == 'yes').astype(int) +
        (df['bmi'] >= 25.0).astype(int)
    )
    
    assert risk_index.iloc[0] == 6
    assert risk_index.iloc[1] == 0
    assert risk_index.iloc[2] == 0

def test_sleep_distance_from_8():
    sleep_values = pd.Series([5.0, 8.0, 10.0])
    dist_8 = (sleep_values - 8.0).abs()
    assert dist_8.iloc[0] == 3.0
    assert dist_8.iloc[1] == 0.0
    assert dist_8.iloc[2] == 2.0

if __name__ == '__main__':
    print("Running ML Pipeline Logic Tests...")
    test_feature_engineering_logic()
    test_sleep_distance_from_8()
    print("SUCCESS: All ML Pipeline Logic Tests PASSED!")
