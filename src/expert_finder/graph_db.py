"""Neo4j connection."""

from __future__ import annotations

import os

import certifi
from neo4j import Driver, GraphDatabase

from expert_finder.config import Settings


def neo4j_driver(settings: Settings) -> Driver:
    # python.org macOS builds ship without a CA bundle, which breaks TLS to Neo4j Aura.
    # Fall back to certifi's bundle unless the environment already names one.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
