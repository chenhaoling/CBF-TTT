import unittest

from scripts.inventory_in_place_ttt_public_corpus import normalize_file, order_files


def metadata(path, size=10):
    return {
        "name": path,
        "size": size,
        "blob_id": "blob-" + path,
        "lfs": {"size": size, "sha256": "a" * 64},
    }


class PublicCorpusInventoryTest(unittest.TestCase):
    def test_longcrawl_uses_numeric_not_lexical_order(self):
        files = []
        for index in reversed(range(256)):
            path = f"train/{index}-of-256.parquet"
            files.append(normalize_file("repo", "rev", path, metadata(path)))
        ordered = order_files("longcrawl", files)
        self.assertEqual(ordered[2]["repository_path"], "train/2-of-256.parquet")
        self.assertEqual(ordered[10]["repository_path"], "train/10-of-256.parquet")
        self.assertEqual(ordered[-1]["repository_path"], "train/255-of-256.parquet")

    def test_longcrawl_requires_every_index(self):
        files = []
        for index in range(255):
            path = f"train/{index}-of-256.parquet"
            files.append(normalize_file("repo", "rev", path, metadata(path)))
        with self.assertRaisesRegex(ValueError, "each shard index"):
            order_files("longcrawl", files)

    def test_fineweb_is_lexicographic_and_scoped(self):
        paths = ["sample/100BT/010.parquet", "sample/100BT/002.parquet"]
        files = [normalize_file("repo", "rev", path, metadata(path)) for path in paths]
        ordered = order_files("fineweb", files)
        self.assertEqual([row["repository_path"] for row in ordered], sorted(paths))
        bad = normalize_file("repo", "rev", "sample/10BT/a.parquet",
                             metadata("sample/10BT/a.parquet"))
        with self.assertRaisesRegex(ValueError, "sample/100BT"):
            order_files("fineweb", [bad])

    def test_invalid_lfs_metadata_is_rejected(self):
        path = "sample/100BT/a.parquet"
        value = metadata(path)
        value["lfs"]["size"] = 9
        with self.assertRaisesRegex(ValueError, "invalid publisher metadata"):
            normalize_file("repo", "rev", path, value)


if __name__ == "__main__":
    unittest.main()
