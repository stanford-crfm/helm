from collections import defaultdict
import dataclasses
from dataclasses import dataclass, field
import datetime
import json
import os
import threading
import time
from typing import Dict, Optional, Any, Callable

def create_usage_dict():
    return defaultdict(Usage)

class AuthenticationError(Exception):
    pass


class InsufficientQuotaError(Exception):
    pass


@dataclass
class Usage:
    """Usage information (for a given user, model group, and granularity)."""
    period: Optional[str] = None  # marks the current period (so we know when to reset `used`)
    used: int = 0
    quota: Optional[int] = None

    def update_period(self, period: str):
        if self.period != period:
            self.period = period
            self.used = 0  # Reset

    def can_use(self):
        return self.quota is None or self.used < self.quota

@dataclass
class User:
    """User information."""
    username: str  # unique
    password: str
    email: str
    fullname: str

    # Usage is tracked and limited at different granularities (each one is model group -> usage information)
    daily: Dict[str, Usage] = field(default_factory=create_usage_dict)
    monthly: Dict[str, Usage] = field(default_factory=create_usage_dict)
    total: Dict[str, Usage] = field(default_factory=create_usage_dict)

    def as_dict(self):
        return {
            'username': self.username,
            'password': self.password,
            'email': self.email,
            'fullname': self.fullname,
            'daily': dict_usage_as_dict(self.daily),
            'monthly': dict_usage_as_dict(self.monthly),
            'total': dict_usage_as_dict(self.total),
        }


@dataclass(frozen=True)
class Authentication:
    """How a request can be authenticated."""
    username: str
    password: str


def dict_usage_from_dict(raw: Dict[str, Any]) -> Dict[str, Usage]:
    result = create_usage_dict()
    for key, value in raw.items():
        result[key] = Usage(**value)
    return result

def dict_usage_as_dict(usages: Dict[str, Usage]) -> Dict[str, Any]:
    return dict((key, dataclasses.asdict(usage)) for key, usage in usages.items())


# Compute the current period.
def compute_daily_period():
    now = datetime.datetime.now()
    return f'{now.year}-{now.month}-{now.day}'
def compute_monthly_period():
    now = datetime.datetime.now()
    return f'{now.year}-{now.month}'
def compute_total_period():
    return 'all'


class Users:
    """Contains information about users."""
    def __init__(self, path: str):
        self.path = path
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
        self.write_thread = threading.Thread(target=write_loop)
        self.write_thread.start()

    def finish(self):
        self.done = True
        self.write_thread.join()
        self.write()

    def read(self):
        with self.global_lock:
            # Read from a file
            self.users = []
            if os.path.exists(self.path):
                for line in open(self.path):
                    # Each line is a user.
                    raw_user = json.loads(line)
                    user = User(
                        username=raw_user['username'],
                        password=raw_user['password'],
                        email=raw_user.get('email'),
                        fullname=raw_user.get('fullname'),
                        daily=dict_usage_from_dict(raw_user.get('daily', {})),
                        monthly=dict_usage_from_dict(raw_user.get('monthly', {})),
                        total=dict_usage_from_dict(raw_user.get('total', {})),
                    )
                    self.users.append(user)

            # Build index
            self.username_to_users = {}
            for user in self.users:
                self.username_to_users[user.username] = user

            self.dirty = False


    def authenticate(self, auth: Authentication):
        """Make sure this is a valid username + password.  Throw exceptions if not."""
        user = self.username_to_users.get(auth.username)
        if not user:
            raise AuthenticationError(f'No such user {auth.username}')
        if user.password != auth.password:
            raise AuthenticationError(f'Incorrect password for user {auth.username}')

    def check_can_use(self, username: str, model_group: str) -> bool:
        """Check if the given `username` can use `model_group`.  Throw exceptions if not."""
        def granular_check_can_use(granularity: str, usages: Dict[str, Usage], model_group: str, compute_period: Callable[[], str]):
            """Helper that checks the usage at a certain granularity (e.g., daily, monthly, total)."""
            usage = usages[model_group]
            period = compute_period()
            usage.update_period(period)
            if not usage.can_use():
                raise InsufficientQuotaError(f'{granularity} quota ({usage.quota}) for {model_group} already used up')

        with self.global_lock:
            user = self.username_to_users[username]
            granular_check_can_use('daily', user.daily, model_group, compute_daily_period)
            granular_check_can_use('monthly', user.monthly, model_group, compute_monthly_period)
            granular_check_can_use('total', user.total, model_group, compute_total_period)

    def use(self, username: str, model_group: str, delta: int):
        """Call this function when user with `username` used `delta` of `model_group`."""
        def granular_use(usages: Dict[str, Usage], model_group: str, compute_period: Callable[[], str]):
            """Helper that checks the usage at a certain granularity (e.g., daily, monthly, total)."""
            usage = usages[model_group]
            period = compute_period()
            usage.update_period(period)
            usage.used += delta

        with self.global_lock:
            user = self.username_to_users[username]
            granular_use(user.daily, model_group, compute_daily_period)
            granular_use(user.monthly, model_group, compute_monthly_period)
            granular_use(user.total, model_group, compute_total_period)
            self.dirty = True


    def write(self):
        """Write what's in memory to disk."""
        with self.global_lock:
            raw_users = []
            for user in self.users:
                raw_users.append(user.as_dict())

            print(f'Writing {len(self.users)} users to {self.path}')
            with open(self.path, 'w') as f:
                for raw_user in raw_users:
                    print(json.dumps(raw_user), file=f)
            self.dirty = False
