import pandas as pd
import numpy as np
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from delta import *
import pyspark
from pyspark.sql.functions import lit


# data_type = ["BehaviorEvents", "FacecamPoses", "SessionMetadata",
#              "UnityFramewise", "UnityTrackwise", "UnityTrialwiseMetrics"]

    
builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
    .config("spark.sql.warehouse.dir", "/mnt/SpatialSequenceLearning/RUN_rYL006/lakehouse/delta_catalog")
    
spark = configure_spark_with_delta_pip(builder).getOrCreate()

lakehouse_type = "SessionMetadata"
lakehouse_folder = f"/mnt/SpatialSequenceLearning/RUN_rYL006/lakehouse/delta_catalog/{lakehouse_type}"

data_path = "/mnt/SpatialSequenceLearning/RUN_rYL006/rYL006_P1100/"
data_files = [file for file in os.listdir(data_path) if "DS_Store" not in file]

for data_file in data_files:
    print("Processing file: ", data_file)
    parquet_path = f"{data_path}/{data_file}/session_analytics/{lakehouse_type}.parquet"

    parquet_df = spark.read.format("parquet").load(parquet_path)
    parquet_df = parquet_df.withColumn("source_file", lit(data_file))

    if not os.path.exists(lakehouse_folder):
        parquet_df.write.format("delta").mode("overwrite").save(lakehouse_folder)
    else:
        # Check if the file has already been processed
        df = spark.read.format("delta").load(lakehouse_folder)
        if df.filter(df.source_file == data_file).count() > 0:
            print("File already processed, skipping")
        else:
            parquet_df.write.format("delta").mode("append").save(lakehouse_folder)

df = spark.read.format("delta").load(lakehouse_folder)
df.show()

# Display the Pandas DataFrame
pd_df = df.toPandas()
print(pd_df)