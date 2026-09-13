from __future__ import annotations

import json
import unittest
import urllib.error

from runpod_guard.api import RunpodAPI, RunpodAPIError


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.value).encode()


class APITests(unittest.TestCase):
    def test_headers_and_list_shape(self):
        seen = {}

        def opener(request, timeout):
            seen["authorization"] = request.get_header("Authorization")
            seen["user_agent"] = request.get_header("User-agent")
            return Response({"data": [{"id": "one"}]})

        api = RunpodAPI("secret", opener=opener)
        self.assertEqual(api.list_pods(), [{"id": "one"}])
        self.assertEqual(seen["authorization"], "Bearer secret")
        self.assertEqual(seen["user_agent"], "runpod-guard/0.1")

    def test_http_error_does_not_include_key(self):
        def opener(_request, timeout):
            raise urllib.error.HTTPError("url", 403, "no", {}, None)

        with self.assertRaises(RunpodAPIError) as caught:
            RunpodAPI("top-secret", opener=opener).list_pods()
        self.assertNotIn("top-secret", str(caught.exception))

    def test_delete_is_confirmed_by_listing(self):
        api = RunpodAPI("secret")
        calls = []
        api.delete_pod = lambda pod_id: calls.append(("delete", pod_id))
        api.list_pods = lambda: []
        self.assertTrue(api.delete_and_confirm("pod", sleeper=lambda _: None))
        self.assertEqual(calls, [("delete", "pod"), ("delete", "pod")])

    def test_unexpected_list_shape_fails_closed(self):
        api = RunpodAPI("secret")
        api.request = lambda *_args: {"unexpected": "shape"}
        with self.assertRaises(RunpodAPIError):
            api.list_pods()
        api.request = lambda *_args: [{}]
        with self.assertRaises(RunpodAPIError):
            api.list_pods()

    def test_timeout_during_delete_is_retried(self):
        api = RunpodAPI("secret")
        calls = []

        def delete(pod_id):
            calls.append(pod_id)
            if len(calls) == 1:
                raise TimeoutError

        api.delete_pod = delete
        api.list_pods = lambda: []
        self.assertTrue(api.delete_and_confirm("pod", sleeper=lambda _: None))
        self.assertEqual(len(calls), 2)

    def test_stop_is_confirmed_by_two_status_reads(self):
        api = RunpodAPI("secret")
        calls = []
        api.stop_pod = lambda pod_id: calls.append(pod_id)
        api.get_pod = lambda _pod_id: {"desiredStatus": "EXITED"}
        self.assertTrue(api.stop_and_confirm("pod", sleeper=lambda _: None))
        self.assertEqual(calls, ["pod", "pod"])


if __name__ == "__main__":
    unittest.main()
