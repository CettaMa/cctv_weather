"""Proxy stream hanya boleh menjangkau host kamera di katalog (pencegahan SSRF)."""
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as flask_app  # noqa: E402

HOST = "https://pelindung.bandung.go.id:3443"
BLOCKED = [
    "http://127.0.0.1:8000/api/status",
    "http://localhost/admin",
    "http://169.254.169.254/latest/meta-data/",
    "https://example.com/a.m3u8",
    "https://example.com/a.ts",
    "http://pelindung.bandung.go.id:3443/video/a.m3u8",            # skema lain
    "https://pelindung.bandung.go.id:22/video/a.m3u8",             # port lain
    "https://pelindung.bandung.go.id.example.com:3443/a.m3u8",
    "https://pelindung.bandung.go.id:3443@127.0.0.1/a.m3u8",       # userinfo
    "https://127.0.0.1\\@pelindung.bandung.go.id:3443/a.m3u8",     # backslash
    "file:///etc/passwd",
    "ftp://pelindung.bandung.go.id:3443/a",
    "//127.0.0.1/a",
]


def response(status=200, content=b"#EXTM3U\n", headers=None):
    res = mock.Mock(status_code=status, content=content, headers=headers or {})
    res.text = content.decode("latin-1")
    return res


class ProxyAllowlistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = flask_app.app.test_client()

    def setUp(self):
        flask_app.STREAM_CACHE.clear()
        patcher = mock.patch.object(flask_app.UPSTREAM_SESSION, "get", return_value=response())
        self.get = patcher.start()
        self.addCleanup(patcher.stop)

    def test_proxy_stream_blocks_foreign_targets(self):
        for url in BLOCKED:
            with self.subTest(url=url):
                res = self.client.get(f"/api/proxy_stream?url={quote(url, safe='')}")
                self.assertEqual(res.status_code, 403)
        self.get.assert_not_called()

    def test_stream_route_cannot_leave_video_base(self):
        for sub in ("http://127.0.0.1:8000/api/status", "https://example.com/a.ts", "../admin",
                    "a/../../admin", "%2e%2e/admin"):
            with self.subTest(sub=sub):
                res = self.client.get(f"/api/stream/{sub}", follow_redirects=True)
                self.assertIn(res.status_code, (403, 404))
        self.get.assert_not_called()
        # garis miring ganda dirapikan Flask menjadi path biasa, tetap di bawah /video/
        self.client.get("/api/stream/%2F%2Fexample.com/a", follow_redirects=True)
        for call in self.get.call_args_list:
            self.assertTrue(call.args[0].startswith(flask_app.UPSTREAM_VIDEO_BASE), call.args[0])

    def test_catalog_camera_still_works(self):
        res = self.client.get("/api/stream/HIKSVISION/Kamera.m3u8")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.get.call_args.args[0], f"{HOST}/video/HIKSVISION/Kamera.m3u8")
        self.assertFalse(self.get.call_args.kwargs["allow_redirects"])
        res = self.client.get(f"/api/proxy_stream?url={quote(HOST + '/video/a.m3u8', safe='')}")
        self.assertEqual(res.status_code, 200)

    def test_redirect_to_foreign_host_is_blocked(self):
        self.get.return_value = response(302, b"", {"Location": "http://127.0.0.1/secret"})
        for path in ("/api/stream/a.m3u8", "/api/stream/a.ts",
                     f"/api/proxy_stream?url={quote(HOST + '/video/b.m3u8', safe='')}"):
            with self.subTest(path=path):
                self.get.reset_mock()
                self.assertEqual(self.client.get(path).status_code, 403)
                self.assertEqual(self.get.call_count, 1)

    def test_redirect_within_catalog_host_is_followed(self):
        self.get.side_effect = [response(302, b"", {"Location": "/video/baru.m3u8"}), response()]
        self.assertEqual(self.client.get("/api/stream/lama.m3u8").status_code, 200)
        self.assertEqual(self.get.call_args.args[0], f"{HOST}/video/baru.m3u8")


if __name__ == "__main__":
    unittest.main()
