import pandas as pd
import numpy as np
import os
from delta import *
import pyspark
from pyspark.sql.functions import lit
import time
from pyspark.sql.functions import col, avg, count, floor
from pyspark.sql import SparkSession
import delta
from pyspark.sql.functions import broadcast


# Set up the Spark session
# Notice here the additional configuration for the Spark driver and executor memory
# since partition is a memory-intensive operation
builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
    .config("spark.sql.warehouse.dir", "/mnt/SpatialSequenceLearning/RUN_rYL006/lakehouse/delta_catalog")
    
spark = configure_spark_with_delta_pip(builder).getOrCreate()

lakehouse_folder = "/mnt/SpatialSequenceLearning/RUN_rYL006/lakehouse/delta_catalog/"
session_name = "2024-11-21_17-22_rYL006_P1100_LinearTrackStop_25min"

unity_table = os.path.join(lakehouse_folder, "UnityFramewise")
unity_df = spark.read.format("delta").load(unity_table)
unity_df_filtered = unity_df.filter(unity_df.source_file == session_name)

start_ephys_time = unity_df_filtered.select("frame_ephys_timestamp").first().frame_ephys_timestamp
time_interval = 33333

# Loop to process data in chunks of 33333 microseconds
while True:
    end_ephys_time = start_ephys_time + time_interval

    start_time = time.time()

    filtered_df = unity_df_filtered.filter(
        (unity_df_filtered.frame_ephys_timestamp >= start_ephys_time) &
        (unity_df_filtered.frame_ephys_timestamp <= end_ephys_time)
    )

    end_time = time.time()
    print(f"Time taken for interval {start_ephys_time} to {end_ephys_time}: {end_time - start_time} seconds")

    # Check if there is no more data to process
    if filtered_df.count() == 0:
        break

    # Update the start time for the next iteration
    start_ephys_time = end_ephys_time

print("done")

