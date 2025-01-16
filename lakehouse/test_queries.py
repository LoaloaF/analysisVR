import pandas as pd
import numpy as np
import os
from delta import *
import pyspark
from pyspark.sql.functions import lit
import time
from pyspark.sql.functions import col, avg, count
from pyspark.sql import SparkSession
import delta
from pyspark.sql.functions import broadcast

def test_lick(lick_df_filtered, unity_df_filtered):

    lick_df_filtered.select("event_ephys_timestamp").show()
    unity_df_filtered.select("frame_ephys_timestamp", "L_count").show()

    # Start timing
    start_time = time.time()

    spark = SparkSession.builder.appName("QueryExample").getOrCreate()

    lick_df_filtered.createOrReplaceTempView("lick_df_filtered")
    unity_df_filtered.createOrReplaceTempView("unity_df_filtered")

    # SQL query to find the data in unity_df_filtered with timestamps within the range and compute the average
    query = """
    SELECT l.event_ephys_timestamp AS lick_timestamp, AVG(u.L_count) AS avg_L_count
    FROM lick_df_filtered l
    JOIN unity_df_filtered u
    ON u.frame_ephys_timestamp BETWEEN l.event_ephys_timestamp - 10000 AND l.event_ephys_timestamp + 10000
    GROUP BY l.event_ephys_timestamp
    """

    # Execute the query
    result_df = spark.sql(query)
    result_df.show()

    # End timing
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

def test_spikes(lick_df_filtered, spike_df_filtered):
    lick_df_filtered.select("event_ephys_timestamp").show()
    spike_df_filtered.select("spike_time").show()


    spark = SparkSession.builder.appName("QueryExample").getOrCreate()

    lick_df_filtered.createOrReplaceTempView("lick_df_filtered")
    spike_df_filtered.createOrReplaceTempView("spike_df_filtered")

    # Start timing
    start_time = time.time()
    # SQL query to find the data in unity_df_filtered with timestamps within the range and compute the average
    # query = """
    # SELECT l.event_ephys_timestamp AS lick_timestamp, COUNT(*) AS num_spikes
    # FROM lick_df_filtered l
    # JOIN spike_df_filtered s
    # ON s.spike_time BETWEEN l.event_ephys_timestamp - 10000 AND l.event_ephys_timestamp + 10000
    # GROUP BY l.event_ephys_timestamp
    # """
    # Execute the query
    # result_df = spark.sql(query)

    # Instead we use the spark built in functions which is faster
    lick_df_filtered = broadcast(lick_df_filtered)

    result_df = lick_df_filtered.alias("l") \
        .join(
            spike_df_filtered.alias("s"),
            (col("s.spike_time") >= col("l.event_ephys_timestamp") - 10000) & 
            (col("s.spike_time") <= col("l.event_ephys_timestamp") + 10000)
        ) \
        .groupBy("l.event_ephys_timestamp") \
        .agg(count("*").alias("num_spikes"))
    
    result_df.show()

    # End timing
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")




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


lick_df = spark.read.format("delta").load(behavior_table)
unity_df = spark.read.format("delta").load(unity_table)
spike_df = spark.read.format("delta").load(spike_table)

# spark.sql(f"OPTIMIZE delta.`{behavior_table}` ZORDER BY (event_pc_timestamp)")
# spark.sql(f"OPTIMIZE delta.`{unity_table}` ZORDER BY (frame_pc_timestamp)")
# spark.sql(f"OPTIMIZE delta.`{spike_table}` ZORDER BY (spike_time)")

lick_df_filtered = lick_df.filter((lick_df.source_file == session_name) & (lick_df.event_name == "L"))
unity_df_filtered = unity_df.filter(unity_df.source_file == session_name)
spike_df_filtered = spike_df.filter(spike_df.source_file == session_name)

test_spikes(lick_df_filtered, spike_df_filtered)