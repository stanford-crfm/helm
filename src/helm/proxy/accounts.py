import copy
import datetime
import random
import string
from typing import Dict, Optional, Callable, List

from dacite import from_dict
from dataclasses import asdict, dataclass, field
from sqlitedict import SqliteDict

from helm.common.authentication import Authentication
from helm.common.general import hlog


# TODO: Move this to a configuration file (`default_quotas.conf` alongside `accounts.jsonl`)
#       https://github.com/stanford-crfm/benchmarking/issues/52
# There is no limit if nothing is specified.
DEFAULT_QUOTAS = {
    # model group -> {granularity -> quota}
    "gpt3": {"daily": 10000},
    "gpt4": {"daily": 10000},
    "codex": {"daily": 10000},
    "jurassic": {"daily": 10000},
    "gooseai": {"daily": 10000},
    "cohere": {"daily": 10000},
    "dall_e": {"daily": 5},  # In terms of the number of generated images
    "together_vision": {"daily": 30},
    "simple": {"daily": 10000},
}


class AuthenticationError(Exception):
    pass


class InsufficientQuotaError(Exception):
    pass


@dataclass
class Usage:
    """Usage information (for a given account, model group, and granularity)."""

    # What period it is (so we know when to reset `used`) - for example, for
    # daily granularity, period might be 2021-12-30
    period: Optional[str] = None

    # How many tokens was used
    used: int = 0

    # How much quota do we have (None means unlimited)
    quota: Optional[int] = None

    def update_period(self, period: str):
        if self.period != period:
            self.period = period
            self.used = 0  # Reset in a new period

    def can_use(self):
        return self.quota is None or self.used < self.quota


@dataclass
class Account:
    """An `Account` provides access to the API."""

    # Unique API key that is used both for authentication and for identification.
    # Like credit card numbers, this is a bit of a shortcut since we're trying
    # to avoid building out a full-blown system.  If an API key needs to be
    # replaced, we can simply change it and keep the other data the same.
    api_key: str

    # What this account is used for (can include the user names)
    description: str = ""

    # Emails associated this account
    emails: List[str] = field(default_factory=list)

    # What groups this account is associated with
    groups: List[str] = field(default_factory=list)

    # Whether this account has admin access (e.g., ability to modify accounts)
    is_admin: bool = False

    # Usage is tracked and limited at different granularities
    # `usages`: model group -> granularity -> Usage
    usages: Dict[str, Dict[str, Usage]] = field(default_factory=dict)


def set_default_quotas(account: Account):
    """Impose the `DEFAULT_QUOTAS` on the `account` if they don't exist, but don't override anything."""
    for model_group, default_quotas in DEFAULT_QUOTAS.items():
        model_group_usages = account.usages.get(model_group)
        if model_group_usages is None:
            model_group_usages = account.usages[model_group] = {}
        for granularity, quota in default_quotas.items():
            usage = model_group_usages.get(granularity)
            if usage is None:
                usage = model_group_usages[granularity] = Usage()
                usage.quota = quota


# Compute the current period associated with now.
def compute_daily_period():
    now = datetime.datetime.now()
    return f"{now.year}-{now.month}-{now.day}"


def compute_monthly_period():
    now = datetime.datetime.now()
    return f"{now.year}-{now.month}"


def compute_total_period():
    return "all"


class Accounts:
    """
    Contains information about accounts.
    `path`: Path to sqlite file where the information about accounts is stored.
    """

    DEFAULT_API_KEY: str = "root"

    def __init__(self, path: str, root_mode=False):
        self.path: str = path

        with SqliteDict(self.path) as cache:
            # If there isn't a single account, create a default account with admin access.
            if len(cache) == 0:
                account = Account(api_key=Accounts.DEFAULT_API_KEY, is_admin=True)
                set_default_quotas(account)

                cache[Accounts.DEFAULT_API_KEY] = asdict(account)
                cache.commit()
                hlog(f"There were no accounts. Created an admin account with API key: {Accounts.DEFAULT_API_KEY}")
            else:
                hlog(f"Found {len(cache)} account(s).")

        self.root_mode: bool = root_mode

    def authenticate(self, auth: Authentication):
        """Make sure this is a valid API key. Throw exception if not."""
        if self.root_mode:
            return

        with SqliteDict(self.path) as cache:
            self._authenticate_with_cache(auth, cache)

    def _authenticate_with_cache(self, auth: Authentication, sqlite_cache: Dict):
        if self.root_mode:
            return

        if auth.api_key not in sqlite_cache:
            raise AuthenticationError(f"Invalid API key {auth.api_key}")

    def check_admin(self, auth: Authentication):
        """Make sure this is an admin account. Throw exception if not."""
        if self.root_mode:
            return

        with SqliteDict(self.path) as cache:
            self._check_admin_with_cache(auth, cache)

    def _check_admin_with_cache(self, auth: Authentication, sqlite_cache: Dict):
        if self.root_mode:
            return

        self._authenticate_with_cache(auth, sqlite_cache)

        account: Account = from_dict(Account, sqlite_cache[auth.api_key])
        if not account.is_admin:
            raise AuthenticationError(f"API key {auth.api_key} does not have admin privileges.")

    def get_account(self, auth: Authentication) -> Account:
        """
        Fetch current user's account.
        """
        with SqliteDict(self.path) as cache:
            self._authenticate_with_cache(auth, cache)
            return from_dict(Account, cache.get(auth.api_key))

    def get_all_accounts(self, auth: Authentication) -> List[Account]:
        """
        Fetch all accounts (admin-only).
        """
        with SqliteDict(self.path) as cache:
            self._check_admin_with_cache(auth, cache)
            return [from_dict(Account, account_dict) for account_dict in cache.values()]

    def create_account(self, auth: Authentication) -> Account:
        """
        Creates a new account with a random API key and returns that account (admin-only).
        """
        with SqliteDict(self.path) as cache:
            self._check_admin_with_cache(auth, cache)

            api_key: str = self._generate_nonexistent_api_key()
            account = Account(api_key=api_key)
            set_default_quotas(account)

            # Write new account to SqliteDict
            cache[api_key] = asdict(account)
            cache.commit()
            return account

    def delete_account(self, auth: Authentication, api_key: str) -> Account:
        """
        Deletes an account (admin-only).
        """
        with SqliteDict(self.path) as cache:
            self._check_admin_with_cache(auth, cache)

            account_dict = cache.get(api_key)
            if not account_dict:
                raise ValueError(f"Account with API key {api_key} does not exist.")

            account: Account = from_dict(Account, account_dict)
            del cache[api_key]
            cache.commit()
            return account

    def rotate_api_key(self, auth: Authentication, account: Account) -> Account:
        """
        Generate a new API key for an account (admin-only).
        """
        with SqliteDict(self.path) as cache:
            self._check_admin_with_cache(auth, cache)

            old_api_key: str = account.api_key
            new_api_key: str = self._generate_nonexistent_api_key()

            account_dict = cache.get(old_api_key)
            if not account_dict:
                raise ValueError(f"Account with API key {old_api_key} does not exist.")

            account = from_dict(Account, account_dict)
            account.api_key = new_api_key
            cache[new_api_key] = asdict(account)
            del cache[old_api_key]
            cache.commit()
            return account

    def _generate_nonexistent_api_key(self):
        def generate_api_key() -> str:
            return "".join(
                random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(32)
            )

        # The chance of generating an API key that already exists is tiny, but be extra safe
        # by checking the API key does not already exist in the database
        api_key: str = generate_api_key()
        with SqliteDict(self.path) as cache:
            while api_key in cache:
                api_key = generate_api_key()
        return api_key

    def update_account(self, auth: Authentication, account: Account) -> Account:
        """
        Update account except `api_key`. Only an admin or the owner of the account can update.
        """
        with SqliteDict(self.path) as cache:
            self._authenticate_with_cache(auth, cache)

            # Check that the account we're updating exists.
            if account.api_key not in cache:
                raise ValueError(f"Account with API key {account.api_key} does not exist.")

            editor: Account = from_dict(Account, cache.get(auth.api_key))
            current_account: Account = from_dict(Account, cache.get(account.api_key))

            if not editor.is_admin and editor.api_key != account.api_key:
                raise AuthenticationError(
                    f"A user with API key {auth.api_key} attempted to edit an account that doesn't belong to them."
                )

            current_account.description = account.description
            current_account.emails = account.emails
            current_account.groups = account.groups

            if editor.is_admin:
                current_account.is_admin = account.is_admin

                # `used` field in any Usage is immutable, so copy current values of used
                usages = copy.deepcopy(account.usages)
                for service_key, service in usages.items():
                    for granularity_key, granularity in service.items():
                        if (
                            service_key in current_account.usages
                            and granularity_key in current_account.usages[service_key]
                        ):
                            current_used: int = current_account.usages[service_key][granularity_key].used
                            usages[service_key][granularity_key].used = current_used
                current_account.usages = usages

            cache[account.api_key] = asdict(current_account)
            cache.commit()
            return current_account

    def check_can_use(self, api_key: str, model_group: str):
        """Check if the given `api_key` can use `model_group`.  Throw exceptions if not."""

        def granular_check_can_use(
            account: Account,
            model_group: str,
            granularity: str,
            compute_period: Callable[[], str],
        ) -> None:
            """Helper that checks the usage at a certain granularity (e.g., daily, monthly, total)."""

            model_group_usages = account.usages.get(model_group)
            if model_group_usages is None:
                # Assume no restrictions
                return

            usage = model_group_usages.get(granularity)
            if usage is None:
                # Assume no restrictions
                return

            period = compute_period()
            usage.update_period(period)
            if not usage.can_use():
                raise InsufficientQuotaError(f"{granularity} quota ({usage.quota}) for {model_group} already used up")

        def check_non_empty_quota(
            account: Account,
            model_group: str,
        ) -> None:
            """Helper that checks that the account has quota at some granularity.

            At each granularity, a quota of None means unlimited quota.
            However, if the quota is None at every granularity, it means that there is no quota.
            To enforce this rule, this helper raises a InsufficientQuotaError if the quota is None
            at every granularity."""
            model_group_usages = account.usages.get(model_group)
            if model_group_usages is None:
                raise InsufficientQuotaError(f"No quota for {model_group}")
            if all(
                [
                    granularity_usage.quota is None or granularity_usage.quota <= 0
                    for granularity_usage in model_group_usages.values()
                ]
            ):
                raise InsufficientQuotaError(f"No quota for {model_group}")

        if self.root_mode:
            return

        with SqliteDict(self.path) as cache:
            account: Account = from_dict(Account, cache[api_key])
        if account.is_admin:
            return
        granular_check_can_use(account, model_group, "daily", compute_daily_period)
        granular_check_can_use(account, model_group, "monthly", compute_monthly_period)
        granular_check_can_use(account, model_group, "total", compute_total_period)
        check_non_empty_quota(account, model_group)

    def use(self, api_key: str, model_group: str, delta: int):
        """
        Updates the usages: account with `api_key` has used `delta` tokens of `model_group`.
        """

        def granular_use(
            account: Account,
            model_group: str,
            granularity: str,
            compute_period: Callable[[], str],
        ):
            """Helper that checks the usage at a certain granularity (e.g., daily, monthly, total)."""
            # Even if usages don't exist, still keep track.
            model_group_usages = account.usages.get(model_group)
            if model_group_usages is None:
                model_group_usages = account.usages[model_group] = {}

            usage = model_group_usages.get(granularity)
            if usage is None:
                usage = model_group_usages[granularity] = Usage()

            period = compute_period()
            usage.update_period(period)
            usage.used += delta

        if self.root_mode:
            return

        with SqliteDict(self.path) as cache:
            account: Account = from_dict(Account, cache[api_key])
            granular_use(account, model_group, "daily", compute_daily_period)
            granular_use(account, model_group, "monthly", compute_monthly_period)
            granular_use(account, model_group, "total", compute_total_period)
            cache[api_key] = asdict(account)
            cache.commit()
