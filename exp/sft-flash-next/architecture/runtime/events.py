"""Stable event envelopes independent of payload field names."""


def event_row(name, elapsed_seconds, payload):
    return {**payload, 'event': name, 'elapsed_seconds': elapsed_seconds}
