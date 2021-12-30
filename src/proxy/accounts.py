from dacite import from_dict
import dataclasses
from dataclasses import dataclass, field
import datetime
import json
import os
import threading
import time
from typing import Dict, Optional, Callable, List

from common.hierarchical_logger import hlog
from common.authentication import Authentication


# TODO: Move this to a configuration file (`default_quotas.conf` alongside `accounts.jsonl`)
# There is no limit if nothing is specified.
DEFAULT_QUOTAS = {
    # model group -> {granularity -> quota}
    "gpt3": {"daily": 10000},
    "codex": {"daily": 10000},
    "jurassic": {"daily": 10000},
}


class AuthenticationError(Exception):
    pass


class InsufficientQuotaError(Exception):
    pass


@dataclass
class Usage:
    """Usage information (for a given account, model group, and granularity)."""

    period: Optional[str] = None  # marks the current period (so we know when to reset `used`)
    used: int = 0
    quota: Optional[int] = None

    def update_period(self, period: str):
        if self.period != period:
            self.period = period
            self.used = 0  # Reset

    def can_use(self):
        # Important: if quota is None, that means there's no quota!
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
    `path`: where the information about accounts is stored.
    If `read_only` is set, don't write to `path`.
    """

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        self.read_only = read_only
        self.global_lock = threading.Lock()

        self.read()

        def write_loop():
            check_period = 0.1
            while True:
                if self.dirty:
                    self.write()
                for _ in range(int(5 / check_period)):
                    if self.done:
                        break
                    time.sleep(check_period)
                if self.done:
                    break

        self.done = False
        if not self.read_only:
            self.write_thread = threading.Thread(target=write_loop)
            self.write_thread.start()

    def finish(self):
        self.done = True
        if not self.read_only:
            self.write_thread.join()
            if self.dirty:
                self.write()

    def read(self):
        with self.global_lock:
            # Read from a file (each line is a JSON file with an account).
            self.accounts = []
            if os.path.exists(self.path):
                for line in open(self.path):
                    account = from_dict(data_class=Account, data=json.loads(line))
                    set_default_quotas(account)
                    self.accounts.append(account)
                hlog(f"Read {len(self.accounts)} accounts from {self.path}")
            else:
                hlog(f"0 accounts since {self.path} doesn't exist")

            # Build index
            self.api_key_to_accounts = {}
            for account in self.accounts:
                self.api_key_to_accounts[account.api_key] = account

            # Set when we have modified self.api_key_to_accounts and need to write to disk.
            self.dirty = False

    def authenticate(self, auth: Authentication):
        """Make sure this is a valid api key.  Throw exceptions if not."""
        if auth.api_key not in self.api_key_to_accounts:
            raise AuthenticationError(f"Invalid API key {auth.api_key}")

    def check_can_use(self, api_key: str, model_group: str):
        """Check if the given `api_key` can use `model_group`.  Throw exceptions if not."""

        def granular_check_can_use(
            account: Account, model_group: str, granularity: str, compute_period: Callable[[], str],
        ):
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

        with self.global_lock:
            account = self.api_key_to_accounts[api_key]
            granular_check_can_use(account, model_group, "daily", compute_daily_period)
            granular_check_can_use(account, model_group, "monthly", compute_monthly_period)
            granular_check_can_use(account, model_group, "total", compute_total_period)

    def use(self, api_key: str, model_group: str, delta: int):
        """Call this function when account with `api_key` used `delta` of `model_group`."""

        def granular_use(
            account: Account, model_group: str, granularity: str, compute_period: Callable[[], str],
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

        with self.global_lock:
            account = self.api_key_to_accounts[api_key]
            granular_use(account, model_group, "daily", compute_daily_period)
            granular_use(account, model_group, "monthly", compute_monthly_period)
            granular_use(account, model_group, "total", compute_total_period)
            self.dirty = True

    def write(self):
        """Write what's in memory to disk."""
        with self.global_lock:
            raw_accounts = []
            for account in self.accounts:
                raw_accounts.append(dataclasses.asdict(account))

            if not self.read_only:
                hlog(f"Writing {len(self.accounts)} accounts to {self.path}")
                with open(self.path, "w") as f:
                    for raw_account in raw_accounts:
                        print(json.dumps(raw_account), file=f)
            self.dirty = False
