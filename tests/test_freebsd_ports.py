import importlib
from pathlib import Path
import tomllib
import unittest


class FreeBsdPortsPackagingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.freebsd_dir = self.root / "packaging" / "freebsd"

    def test_wune_port_files_exist_and_are_valid(self):
        wune_port = self.freebsd_dir / "ports" / "audio" / "wune"
        makefile = (wune_port / "Makefile").read_text(encoding="utf-8")
        self.assertIn("PORTNAME=\twune", makefile)
        self.assertIn("DISTVERSION=\t1.2.1.s20261024", makefile)
        self.assertIn("CATEGORIES=\taudio", makefile)
        self.assertIn("USE_GITHUB=\tyes", makefile)
        self.assertIn("GH_ACCOUNT=\tyshdsnd", makefile)
        self.assertIn("GH_PROJECT=\twune", makefile)
        self.assertIn("GH_TAGNAME=\t9e49cd3", makefile)
        self.assertIn("NO_CHECKSUM=\tyes", makefile)
        self.assertIn("WRKSRC=\t\t${.CURDIR}/../../../../..", makefile)
        self.assertIn("USE_PYTHON=\tautoplist distutils", makefile)
        self.assertIn("py-game", makefile)
        self.assertIn("py-soundcard", makefile)
        self.assertNotIn("PAVUCONTROL", makefile)
        self.assertNotIn("pavucontrol", makefile)
        self.assertIn("share/applications/wune.desktop", makefile)
        self.assertIn("share/pixmaps/wune.png", makefile)
        self.assertIn("%%PYTHON_SITELIBDIR%%/wune/build-info.json", makefile)
        self.assertIn("build-info.json", makefile)

        distinfo = (wune_port / "distinfo").read_text(encoding="utf-8")
        self.assertIn("SHA256 (yshdsnd-wune-1.2.1.s20261024-9e49cd3_GH0.tar.gz) = de2f0e2a172ab22481440750371c434002460dc94ca123a1f22e8829e083a8d4", distinfo)
        self.assertIn("SIZE (yshdsnd-wune-1.2.1.s20261024-9e49cd3_GH0.tar.gz) = 2470288", distinfo)

        descr = (wune_port / "pkg-descr").read_text(encoding="utf-8")
        self.assertGreater(len(descr), 20)

        desktop = (wune_port / "files" / "wune.desktop").read_text(encoding="utf-8")
        self.assertIn("[Desktop Entry]", desktop)
        self.assertIn("Exec=wune", desktop)
        self.assertIn("Icon=wune", desktop)

    def test_soundcard_port_files_exist_and_match_upstream_release(self):
        sc_port = self.freebsd_dir / "ports" / "audio" / "py-soundcard"
        makefile = (sc_port / "Makefile").read_text(encoding="utf-8")
        self.assertIn("PORTNAME=\tsoundcard", makefile)
        self.assertIn("PORTVERSION=\t0.4.6", makefile)
        self.assertIn("MASTER_SITES=\tPYPI", makefile)
        self.assertIn("USE_PYTHON=\tautoplist distutils", makefile)

        distinfo = (sc_port / "distinfo").read_text(encoding="utf-8")
        self.assertIn("SHA256 (soundcard-0.4.6.tar.gz) = 9b46d64a26f97cd7d88bcfc385c97c069f97c5ec3404e4e7c23771598a8cf47b", distinfo)
        self.assertIn("SIZE (soundcard-0.4.6.tar.gz) = 40830", distinfo)

        descr = (sc_port / "pkg-descr").read_text(encoding="utf-8")
        self.assertIn("SoundCard", descr)

    def test_helper_makefile_and_documentation_exist(self):
        helper = (self.freebsd_dir / "Makefile").read_text(encoding="utf-8")
        self.assertIn("package-soundcard", helper)
        self.assertIn("package-wune", helper)

        readme = (self.freebsd_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("pkg add", readme)
        self.assertIn("/usr/ports/audio/", readme)
        self.assertIn("9e49cd3", readme)

    def test_pyproject_and_setup_metadata(self):
        pyproject_content = (self.root / "pyproject.toml").read_bytes()
        data = tomllib.loads(pyproject_content.decode("utf-8"))
        self.assertEqual(data["project"]["name"], "wune")
        self.assertEqual(data["project"]["version"], "1.2.1")
        self.assertEqual(data["project"]["scripts"]["wune"], "wune.cli:main")

        setup_content = (self.root / "setup.py").read_text(encoding="utf-8")
        self.assertIn('name="wune"', setup_content)
        self.assertIn('version="1.2.1"', setup_content)
        self.assertIn('"wune = wune.cli:main"', setup_content)

    def test_cli_entry_point_imports_cleanly(self):
        import wune.cli as cli
        self.assertTrue(callable(cli.main))


if __name__ == "__main__":
    unittest.main()
