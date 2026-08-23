from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crawl_kinh_thi.py"
SPEC = importlib.util.spec_from_file_location("crawl_kinh_thi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


GROUP_HTML = """
<h4 class="poem-group-title">
  <a href="/Khong-Tu/Quoc-phong/group-section_1">Quốc phong</a>
</h4>
<div class="poem-group-list"><ol>
  <li><a href="/Khong-Tu/An-ky-loi-1/poem-test_UID-1">Ẩn kỳ lôi 1</a></li>
  <li><a href="/Khong-Tu/An-ky-loi-2/poem-test_UID-2">Ẩn kỳ lôi 2</a></li>
</ol></div>
"""

POEM_HTML = """
<div class="poem-content">
  <div class="poem-view-separated">
    <h4><strong class="han-chinese">殷其雷 1</strong></h4>
    <p class="han-chinese">殷其雷，<br/>在南山之陽。</p>
    <p>&nbsp;</p>
    <h4><strong>Ẩn kỳ lôi 1</strong></h4>
    <p><span class="popup-comment">Ẩn<sup>[1]</sup></span> kỳ lôi,<br/>Tại nam sơn chi <span>dương</span>.</p>
  </div>
  <h4><strong>Dịch nghĩa</strong></h4>
  <p>Tiếng sấm ầm ầm.</p>
</div>
"""


class CrawlKinhThiTest(unittest.TestCase):
    def test_parse_group_sections_and_poems(self) -> None:
        self.assertEqual(
            MODULE.parse_section_urls(GROUP_HTML),
            ["https://www.thivien.net/Khong-Tu/Quoc-phong/group-section_1"],
        )
        self.assertEqual(
            MODULE.parse_group_poem_urls(GROUP_HTML),
            [
                "https://www.thivien.net/Khong-Tu/An-ky-loi-1/poem-test_UID-1",
                "https://www.thivien.net/Khong-Tu/An-ky-loi-2/poem-test_UID-2",
            ],
        )

    def test_parse_poem_excludes_translation(self) -> None:
        url = "https://www.thivien.net/Khong-Tu/An-ky-loi-1/poem-test_UID-1"
        poem = MODULE.parse_poem(POEM_HTML, url)
        self.assertEqual(poem.title_vi, "Ẩn kỳ lôi 1")
        self.assertEqual(poem.title_ch, "殷其雷 1")
        self.assertEqual(poem.lines_vi, ["Ẩn kỳ lôi,", "Tại nam sơn chi dương."])
        self.assertEqual(poem.lines_ch, ["殷其雷，", "在南山之陽。"])
        self.assertNotIn("Tiếng sấm", poem.lines_vi)

    def test_filename_and_parallel_csv_output(self) -> None:
        poem = MODULE.Poem(
            uid="test",
            url="https://example.test/poem-test",
            title_vi="Ẩn kỳ lôi 1",
            title_ch="殷其雷 1",
            lines_vi=["Ẩn kỳ lôi,"],
            lines_ch=["殷其雷，"],
        )
        self.assertEqual(MODULE.filename_stem(poem.title_vi), "ankyloi1")
        with tempfile.TemporaryDirectory() as directory:
            output_path = MODULE.write_poem(poem, Path(directory), "ankyloi1", False)
            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(output_path.name, "ankyloi1.csv")
            self.assertEqual(rows, [{"vi": "Ẩn kỳ lôi,", "ch": "殷其雷，"}])

    def test_long_filename_is_truncated_with_uid(self) -> None:
        stem = MODULE.filename_stem("Bạch vân " * 100, "test_UID-123")
        self.assertLessEqual(len(stem), MODULE.MAX_FILENAME_STEM_LENGTH)
        self.assertTrue(stem.endswith("-test_UID-123"))

        poem = MODULE.Poem(
            uid="test_UID-123",
            url="https://example.test/poem-test_UID-123",
            title_vi="Bạch vân " * 100,
            title_ch="白雲",
            lines_vi=["Bạch vân"],
            lines_ch=["白雲"],
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = MODULE.write_poem(poem, Path(directory), stem, False)
            self.assertTrue(output_path.exists())
            self.assertLessEqual(len(output_path.name.encode("utf-8")), 255)


if __name__ == "__main__":
    unittest.main()
