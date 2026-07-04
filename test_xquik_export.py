import unittest

from xquik_export import build_xquik_metadata, normalize_xquik_export


class XquikExportTests(unittest.TestCase):
    def test_normalizes_common_export_columns(self):
        self.assertEqual(
            normalize_xquik_export(
                "tweet_id,full_text,author_username,created_at\n"
                "42,Great launch,@maker,2026-07-04\n"
            ),
            [
                {
                    "comment": "Great launch",
                    "username": "@maker",
                    "source_id": "42",
                    "created_at": "2026-07-04",
                }
            ],
        )

    def test_skips_blank_text_rows_and_falls_back_to_row_id(self):
        rows = normalize_xquik_export(
            "text,user\n"
            "   ,empty\n"
            "Useful signal,analyst\n"
        )

        self.assertEqual(
            rows,
            [
                {
                    "comment": "Useful signal",
                    "username": "analyst",
                    "source_id": "2",
                    "created_at": "",
                }
            ],
        )

    def test_builds_dashboard_metadata_from_import(self):
        metadata = build_xquik_metadata([{"comment": "one"}])

        self.assertEqual(metadata["totalComment"], 1)


if __name__ == "__main__":
    unittest.main()
