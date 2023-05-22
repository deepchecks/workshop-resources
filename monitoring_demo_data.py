import pandas as pd
import numpy as np

# Load csv - https://drive.google.com/file/d/1oTzW6p52afTIJ4GGWhEi970tEYmINkid/view
data = pd.read_csv('airbnb.csv')
print(data.head())
# === Metadata ====
categorical_features = ['room_type', 'neighbourhood', 'neighbourhood_group', 'has_availability']
numerical_features = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
                      'availability_365']
label = 'price'
prediction = 'predictions'

# Split into reference and production
reference = data.sample(frac=0.2, random_state=200)
production = data.drop(reference.index)

# Create timestamps for production
current_time = int(time.time())
time_test_start = current_time - 86400 * 30  # Span data for 30 days
timestamps = np.sort((np.random.rand(len(production)) * (current_time - time_test_start)) + time_test_start)
