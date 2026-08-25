import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crawl_that_ngon_bat_cu.py"
SPEC = importlib.util.spec_from_file_location("crawl_that_ngon_bat_cu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self):
        self.urls = []

    def get_text(self, url):
        self.urls.append(url)
        return "small-partition-html"


class ShallowCrawlerTests(unittest.TestCase):
    def test_pipeline_scope_is_country_2_and_selected_ages(self):
        crawler = MODULE.configure_crawler(MODULE.load_crawler())
        self.assertEqual(("2",), crawler.DEFAULT_COUNTRY_IDS)
        self.assertEqual({"2": MODULE.AGE_COUNTS}, crawler.DEFAULT_SHALLOW_AGE_PARTITION_COUNTS)

    def test_shallow_strategy_uses_all_small_pages_and_date_windows(self):
        crawler = MODULE.configure_crawler(MODULE.load_crawler())
        client = FakeClient()
        small_calls = []
        window_calls = []

        def fake_small(_client, _html, base_url, label):
            small_calls.append((base_url, label))
            return ["https://example.test/poem-small"]

        def fake_window(_client, base_url, label):
            window_calls.append((base_url, label))
            return [f"https://example.test/{label}"]

        crawler.collect_small_partition = fake_small
        crawler.collect_capped_window = fake_window
        with tempfile.TemporaryDirectory() as directory:
            urls = crawler.collect_shallow_age_windows(
                client,
                Path(directory) / "urls.json",
                False,
                {"2": {"52": 3, "53": 347}},
                ["2"],
            )

        self.assertEqual(1, len(small_calls))
        self.assertEqual(2, len(window_calls))
        self.assertIn("Country=2&Age%5B%5D=52", small_calls[0][0])
        self.assertIn("&Sort=Date", window_calls[0][0])
        self.assertNotIn("SortOrder=desc", window_calls[0][0])
        self.assertIn("&Sort=Date&SortOrder=desc", window_calls[1][0])
        self.assertEqual(3, len(urls))

    def test_rejects_unconfigured_country(self):
        crawler = MODULE.configure_crawler(MODULE.load_crawler())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(crawler.CrawlError, "không cấu hình country ID"):
                crawler.collect_shallow_age_windows(
                    FakeClient(),
                    Path(directory) / "urls.json",
                    False,
                    {"2": {"52": 3}},
                    ["3"],
                )

    def test_limit_is_a_stable_sample_across_discovered_urls(self):
        crawler = MODULE.configure_crawler(MODULE.load_crawler())
        discovered = [f"https://example.test/poem-{index}" for index in range(20)]
        crawler.collect_shallow_age_windows = lambda *args, **kwargs: discovered

        first = crawler.collect_urls(FakeClient(), 5, Path("unused.json"), False, ["2"])
        expanded = crawler.collect_urls(FakeClient(), 10, Path("unused.json"), False, ["2"])

        self.assertEqual(first, expanded[:5])
        self.assertNotEqual(discovered[:5], first)


if __name__ == "__main__":
    unittest.main()
