import json

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


repository = "HuggingFaceFW/fineweb-edu"
revision = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
publisher_file = "sample/10BT/000_00000.parquet"
uri = f"hf://datasets/{repository}@{revision}/{publisher_file}"
filesystem = HfFileSystem()
info = filesystem.info(uri)
print(json.dumps({"uri": uri, "info": info}, default=str, sort_keys=True), flush=True)
with filesystem.open(uri, "rb") as source:
    parquet = pq.ParquetFile(source)
    print(json.dumps({"rows": parquet.metadata.num_rows,
                      "row_groups": parquet.metadata.num_row_groups,
                      "schema": parquet.schema.names}, sort_keys=True), flush=True)
    batch = next(parquet.iter_batches(batch_size=2, columns=["text"]))
    print(json.dumps({"text_lengths": [len(row["text"]) for row in batch.to_pylist()]},
                     sort_keys=True), flush=True)
print("clean_exit", flush=True)
