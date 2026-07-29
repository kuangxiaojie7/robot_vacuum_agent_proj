import os
import unittest
from unittest.mock import patch

from agent.tools import agent_tools
from utils.config_handler import agent_conf


class GaodeApiKeyConfigTests(unittest.TestCase):
    def test_agent_config_does_not_store_gaode_key(self):
        self.assertNotIn("gaodekey", agent_conf)

    def test_gaode_request_requires_environment_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GAODE_API_KEY"):
                agent_tools._gaode_get("/v3/ip", {})


if __name__ == "__main__":
    unittest.main()
