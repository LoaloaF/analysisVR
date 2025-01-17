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
unity_df = spark.read.format("delta").load(unity_table)



