import os
import pytest
import tempfile

from helm.proxy.accounts import Accounts, Authentication, InsufficientQuotaError, Usage


class TestAutoTokenCounter:
    def setup_method(self, method):
        accounts_file = tempfile.NamedTemporaryFile(delete=False)
        self.accounts_path: str = accounts_file.name
        self.accounts = Accounts(self.accounts_path)
        self.root_auth = Authentication(Accounts.DEFAULT_API_KEY)

    def teardown_method(self, method):
        os.remove(self.accounts_path)

    def test_check_can_use(self):
        model_group = "anthropic"
        account = self.accounts.create_account(self.root_auth)

        # Cannot use this account because no quota was added
        with pytest.raises(InsufficientQuotaError):
            self.accounts.check_can_use(account.api_key, model_group)

        # Add monthly quota
        account.usages[model_group] = {}
        account.usages[model_group]["monthly"] = Usage(quota=1000)
        self.accounts.update_account(self.root_auth, account)

        # Now this account has quota and can be used
        self.accounts.check_can_use(account.api_key, model_group)
