"""
Script for counting how many porn themed texts there are at each domain.
Saves a CSV file containing the domains and the counts associated with them.

Domains that do not contain porn at all will not be saved.
"""

import os

import pandas as pd

from utils.streams import chunk, stream_all_records, to_text_stream
from utils.topic import PornClassifier

TOPIC_MODEL = "nmf_100"
TOPIC_MODEL_PATH = "/work/topic_model/"

DATA_PATH = "/work/netarkivet-cleaned/"
SAVE_PATH = f"/work/porn_domains/{TOPIC_MODEL}_porn_counts.csv"


def main():
    """
    Main function of the script
    """
    print("----Loading porn classifier----")
    cls = PornClassifier.load(TOPIC_MODEL, TOPIC_MODEL_PATH)
    print("...")
    print("----Setting up data stream-----")
    records = stream_all_records(DATA_PATH)
    # Chunking records, so that if something goes south we still got some counts saved
    record_chunks = chunk(records, 150_000)
    total_counts = pd.DataFrame({})
    print("...")
    for index, record_chunk in enumerate(record_chunks):
        print("START:")
        # I can totally do this as record_chunk is a list
        # I won't exhaust the stream
        texts = to_text_stream(record_chunk)
        domains = (record["domain_key"] for record in record_chunk)
        # Obtaining predictions with the topic-based porn classifier
        predictions = cls.predict(texts)
        # Using pandas' routines to count True predictions for each domain
        current_counts = (
            pd.DataFrame({"domains": domains, "porn_counts": predictions})
            .groupby("domains")
            .sum()
        )
        # Clear out console
        os.system("clear")
        # Log results of current chunk
        print(f"\n\nChunk no. {index} yielded the following results:")
        print(current_counts)
        # Adding current counts to total counts
        total_counts = total_counts.add(current_counts, fill_value=0).astype(int)
        # Persisting counts to disk
        total_counts.to_csv(SAVE_PATH)


if __name__ == "__main__":
    main()
