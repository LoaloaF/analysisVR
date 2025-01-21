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

behavior_table = os.path.join(lakehouse_folder, "BehaviorEvents")
unity_table = os.path.join(lakehouse_folder, "UnityFramewise")
metadata_table = os.path.join(lakehouse_folder, "SessionMetadata")
spike_table = os.path.join(lakehouse_folder, "Spikes")

behavior_df = spark.read.format("delta").load(behavior_table)
spike_df = spark.read.format("delta").load(spike_table)

# create partition on spike table into minutes
spike_df_partitioned = spike_df.withColumn(
    "spike_time_minute", floor((col("spike_time") / 1_000_000).cast("long") / 60)  # Convert microminutes to minutes
)

# Write the spike data into a partitioned Delta table
spike_df_partitioned.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("spike_time_minute") \
    .save(spike_table)

# Optimize the Delta table to reduce small files
delta_table = DeltaTable.forPath(spark, spike_table)
delta_table.optimize()

# Similar as before, but on behavior tables
behavior_df_partitioned = behavior_df.withColumn(
    "event_ephys_minute", floor((col("event_ephys_timestamp") / 1_000_000).cast("long") / 60)  # Convert microminutes to minutes
)

behavior_df_partitioned.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("event_ephys_minute") \
    .save(behavior_table)

delta_table = DeltaTable.forPath(spark, behavior_table)
delta_table.optimize()