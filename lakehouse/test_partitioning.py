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



builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
    .config("spark.sql.warehouse.dir", "/mnt/SpatialSequenceLearning/RUN_rYL006/lakehouse/delta_catalog")
    
spark = configure_spark_with_delta_pip(builder).getOrCreate()

lakehouse_folder = "/mnt/SpatialSequenceLearning/RUN_rYL006/lakehouse/delta_catalog/"
session_name = "2024-11-21_17-22_rYL006_P1100_LinearTrackStop_25min"

behavior_table = os.path.join(lakehouse_folder, "BehaviorEvents")
unity_table = os.path.join(lakehouse_folder, "UnityFramewise")
metadata_table = os.path.join(lakehouse_folder, "SessionMetadata")
spike_table = os.path.join(lakehouse_folder, "Spikes")

# delta.DeltaTable.forPath(spark, behavior_table).optimize().executeZOrderBy("event_pc_timestamp")
# delta.DeltaTable.forPath(spark, spike_table).optimize().executeZOrderBy("spike_time")
behavior_df = spark.read.format("delta").load(behavior_table)
spike_df = spark.read.format("delta").load(spike_table)


spike_df_partitioned = spike_df.withColumn(
    "spike_time_second", floor(col("spike_time") / 1_000_000)  # Convert microseconds to seconds
)

# Write the spike data into a partitioned Delta table
spike_df_partitioned.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("spike_time_second") \
    .save(spike_table)

behavior_df_partitioned = behavior_df.withColumn(
    "event_ephys_second", floor(col("event_ephys_timestamp") / 1_000_000)  # Convert microseconds to seconds
)

# Write the spike data into a partitioned Delta table
behavior_df_partitioned.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("event_ephys_second") \
    .save(behavior_table)