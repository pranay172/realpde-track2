"""Synthetic regression tests for the public-release checks."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_archive import DERIVED_FNO_FILES, check_derived_licenses, check_document_links

SPEC = importlib.util.spec_from_file_location(
    "snapshot_audit", Path(__file__).resolve().parents[1] / "scripts/publication_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PublicationChecks(unittest.TestCase):
    def test_html_links_accept_attribute_quotes_and_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target.svg").write_text("<svg />")
            document = root / "README.md"
            for attribute in ("src", "href", "SRC", "HREF"):
                for quote in ('"', "'", ""):
                    with self.subTest(attribute=attribute, quote=quote):
                        document.write_text(f"<img {attribute} = {quote}target.svg{quote} />")
                        self.assertEqual(check_document_links(document), 1)

    def test_missing_html_links_fail_for_both_quote_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "README.md"
            for attribute in ("src", "href"):
                for quote in ('"', "'"):
                    with self.subTest(attribute=attribute, quote=quote):
                        document.write_text(f"<a {attribute}={quote}missing.svg{quote}>x</a>")
                        with self.assertRaisesRegex(ValueError, "Missing reader-doc link"):
                            check_document_links(document)

    def test_links_resolve_relative_to_nested_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "docs"
            nested.mkdir()
            (root / "LICENSE").write_text("Synthetic license fixture\n")
            (nested / "figure & preview.svg").write_text("<svg />")
            document = nested / "README.md"
            document.write_text(
                '[License](../LICENSE#scope)\n'
                '<img src="figure%20&amp;%20preview.svg?raw=1#panel" />\n'
                '[Preview](figure%20%26%20preview.svg?raw=1#panel)\n'
            )
            self.assertEqual(check_document_links(document), 3)

    def test_external_links_and_same_page_anchors_are_not_file_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "README.md"
            document.write_text(
                '[Remote](https://example.org/missing.md)\n'
                '<img src="//example.org/missing.svg" />\n'
                '<a href="mailto:realpde-competition@googlegroups.com">Contact</a>\n'
                '[Section](#section)\n<a href="?plain=1#section">View</a>\n'
            )
            self.assertEqual(check_document_links(document), 0)

    def test_markdown_links_still_reject_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "README.md"
            document.write_text('[Missing](missing.md#section)\n')
            with self.assertRaisesRegex(ValueError, "Missing reader-doc link"):
                check_document_links(document)

    def _license_fixture(self, root):
        for name in DERIVED_FNO_FILES:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            header = "\n".join((ROOT / name).read_text().splitlines()[:4])
            destination.write_text(header + "\n")
        (root / "LICENSING.md").write_text((ROOT / "LICENSING.md").read_text())

    def test_all_five_derived_files_have_notices(self):
        self.assertEqual(check_derived_licenses(ROOT), 5)
        self.assertEqual(set(DERIVED_FNO_FILES), {
            "src/realpde_t2/dual_head_fno.py", "src/realpde_t2/variance_head_fno.py",
            "submission/e021/submission.py", "submission/e024/submission.py",
            "submission/e029/submission.py",
        })

    def test_each_derived_header_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in DERIVED_FNO_FILES:
                with self.subTest(path=name):
                    self._license_fixture(root)
                    (root / name).write_text("# No attribution in synthetic fixture\n")
                    with self.assertRaisesRegex(ValueError, "license/attribution header"):
                        check_derived_licenses(root)

    def test_each_derived_inventory_entry_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in DERIVED_FNO_FILES:
                with self.subTest(path=name):
                    self._license_fixture(root)
                    inventory = root / "LICENSING.md"
                    inventory.write_text(inventory.read_text().replace(f"- `{name}`", ""))
                    with self.assertRaisesRegex(ValueError, "inventory entry"):
                        check_derived_licenses(root)

    def test_source_only_folder_passes_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Research snapshot\n")
            report = AUDIT.audit(root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["history"]["state"], "not_initialized")

    def test_secret_report_is_redacted(self):
        token = b"ghp_" + b"x" * 36
        report = AUDIT.scan_blob("example.py", token)
        self.assertTrue(any(x["category"] == "github_token" for x in report))
        self.assertNotIn(token.decode(), str(report))

    def test_private_path_and_email_are_flagged(self):
        value = (b"/" + b"home" + b"/person/project/ " + b"private" + b"@example.com")
        categories = {x["category"] for x in AUDIT.scan_blob("note.md", value)}
        self.assertIn("personal_path", categories)
        self.assertIn("unreviewed_email", categories)

    def test_organizer_and_artifact_payloads_are_forbidden(self):
        for name in ("realpde_t1_starting_kit_v9/scoring.py",
                     "realpde_t2_starting_kit_v6/local_eval.py",
                     "competition_docs/Terms.md", "model.pth", "sample.h5"):
            self.assertTrue(AUDIT.forbidden_path(name), name)

    def test_data_configuration_is_not_a_dataset(self):
        self.assertFalse(AUDIT.forbidden_path("configs/data/release_v1.json"))

    def test_fresh_git_and_reachable_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git("init", "-b", "main")
            git("config", "user.name", "Snapshot test fixture")
            git("config", "user.email", "pranayvandanapu2001@gmail.com")
            git("config", "commit.gpgsign", "false")
            git("config", "core.hooksPath", str(root / "no-hooks"))
            self.assertEqual(AUDIT.audit(root)["history"]["state"], "fresh_uncommitted")
            (root / "README.md").write_text("Synthetic fixture\n")
            git("add", "README.md")
            git("commit", "-m", "Synthetic safe snapshot")
            self.assertEqual(AUDIT.audit(root)["status"], "passed")
            (root / "removed.txt").write_text("ghp_" + "x" * 36)
            git("add", "removed.txt")
            git("commit", "-m", "Synthetic credential fixture")
            (root / "removed.txt").unlink()
            git("add", "-u")
            git("commit", "-m", "Remove synthetic fixture")
            report = AUDIT.audit(root)
            self.assertTrue(any(x["category"] == "github_token" and "object" in x
                                for x in report["findings"]))
            git("config", "user.email", "unapproved" + "@example.com")
            git("commit", "--allow-empty", "-m", "Synthetic unapproved identity")
            report = AUDIT.audit(root)
            self.assertTrue(any(x["category"] == "unreviewed_commit_email"
                                for x in report["findings"]))

    def test_public_email_requires_noreply_or_exact_approval(self):
        self.assertTrue(AUDIT.public_email(b"123+researcher@users.noreply.github.com"))
        self.assertTrue(AUDIT.public_email(b"pranayvandanapu2001@gmail.com"))
        self.assertEqual(AUDIT.scan_blob("approved.txt", b"pranayvandanapu2001@gmail.com"), [])
        self.assertFalse(AUDIT.public_email(b"another.person" + b"@gmail.com"))
        self.assertFalse(AUDIT.public_email(b"person" + b"@example.com"))

    def test_symlink_is_flagged_without_reading_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "link").symlink_to(root / "absent")
            report = AUDIT.audit(root)
            self.assertTrue(any(x["category"] == "symbolic_link" for x in report["findings"]))


if __name__ == "__main__":
    unittest.main()
