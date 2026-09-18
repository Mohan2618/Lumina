import unittest

from flask import Flask, session

from lumina.utils.image import check_guest_limit, consume_guest_limit


class GuestLimitTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"

    def test_guest_check_does_not_consume_quota(self):
        with self.app.test_request_context("/"):
            allowed, error = check_guest_limit()
            self.assertTrue(allowed)
            self.assertIsNone(error)
            self.assertNotIn("guest_messages", session)

    def test_guest_quota_is_consumed_only_after_success(self):
        with self.app.test_request_context("/"):
            consume_guest_limit()
            self.assertEqual(session["guest_messages"], 1)
            allowed, error = check_guest_limit()
            self.assertTrue(allowed)
            self.assertIsNone(error)

    def test_guest_limit_blocks_after_five_successful_requests(self):
        with self.app.test_request_context("/"):
            for _ in range(5):
                consume_guest_limit()
            allowed, error = check_guest_limit()
            self.assertFalse(allowed)
            self.assertIn("Guest limit reached", error)


if __name__ == "__main__":
    unittest.main()
