from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, avg
import os

# Initialize Spark session
spark = SparkSession.builder.appName("Anxiety Attack Data Analysis").getOrCreate()

# Load the dataset using OS-independent path
file_path = os.path.join(os.getcwd(), "anxiety_attack_dataset.csv")
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 1️⃣ Display Data Details
df.show(truncate=False)        # Show full content without truncation
df.printSchema()               # Display schema
df.describe().show()           # Basic statistical summary
df.summary().show()            # Extended statistical summary

# 2️⃣ Partition Strategy 1: Hash Partitioning (based on 'Category')
hash_partitioned_df = df.repartition(4, col("Category"))  # Repartition into 4 partitions
print(f"Hash Partitioning: {hash_partitioned_df.rdd.getNumPartitions()} partitions")

# Apply Transformation Pipeline on Hash Partitioned Data
hash_transformed_df = (hash_partitioned_df
                        .filter(col("Location") == "Library")      # Filter rows
                        .groupBy("Category")                        # Group by category
                        .agg(sum("Quantity").alias("Total_Items"), # Aggregate
                             avg("Quantity").alias("Avg_Items"))
                        .orderBy(col("Total_Items").desc()))        # Sort by total items

# Show Hash Partitioned Results
hash_transformed_df.show()

# 3️⃣ Partition Strategy 2: Range Partitioning (based on 'Quantity')
range_partitioned_df = df.sortWithinPartitions("Quantity")  # Sort within partitions
print(f"Range Partitioning: {range_partitioned_df.rdd.getNumPartitions()} partitions")

# Apply Transformation Pipeline on Range Partitioned Data
range_transformed_df = (range_partitioned_df
                          .filter(col("Quantity") > 10)           # Filter rows
                          .groupBy("Location")                     # Group by location
                          .agg(count("Item_Name").alias("Item_Count")) # Count items
                          .orderBy(col("Item_Count").desc()))         # Sort by item count

# Show Range Partitioned Results
range_transformed_df.show()

# 4️⃣ Check Partition Sizes (Optional Debugging)
def print_partition_sizes(df):
    partition_sizes = df.rdd.glom().map(len).collect()
    print(f"Partition sizes: {partition_sizes}")

print_partition_sizes(hash_partitioned_df)
print_partition_sizes(range_partitioned_df)

# Stop Spark session
spark.stop()