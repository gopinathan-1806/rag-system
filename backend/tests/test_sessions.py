import time

import pytest

from app.errors import LimitError, NotFoundError
from app.sessions import SessionStore


def test_ttl_eviction() -> None:
    store = SessionStore(ttl_s=0, max_sessions=10)
    s = store.create()
    time.sleep(0.01)
    with pytest.raises(NotFoundError):
        store.get(s.id)


def test_capacity_limit() -> None:
    store = SessionStore(ttl_s=100, max_sessions=1)
    store.create()
    with pytest.raises(LimitError):
        store.create()


def test_session_ids_are_unguessable_and_unique() -> None:
    store = SessionStore(ttl_s=100, max_sessions=10)
    a, b = store.create(), store.create()
    assert a.id != b.id and len(a.id) >= 30
