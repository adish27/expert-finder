"""Load the home-services dataset (raw files only; never ground_truth/)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Dataset:
    businesses: list[dict]
    experts: list[dict]
    offerings: list[dict]
    customers: list[dict]
    estimates: list[dict]
    reviews: list[dict]
    posts: list[dict]
    services: list[dict]
    categories: list[dict]

    @classmethod
    def load(cls, data_dir: Path) -> Dataset:
        ref = data_dir / "reference"
        return cls(
            businesses=read_jsonl(data_dir / "businesses.jsonl"),
            experts=read_jsonl(data_dir / "experts.jsonl"),
            offerings=read_jsonl(data_dir / "service_offerings.jsonl"),
            customers=read_jsonl(data_dir / "customers.jsonl"),
            estimates=read_jsonl(data_dir / "estimates.jsonl"),
            reviews=read_jsonl(data_dir / "reviews.jsonl"),
            posts=read_jsonl(data_dir / "social_posts.jsonl"),
            services=json.loads((ref / "services.json").read_text()),
            categories=json.loads((ref / "categories.json").read_text()),
        )

    @cached_property
    def business_by_id(self) -> dict[str, dict]:
        return {b["business_id"]: b for b in self.businesses}

    @cached_property
    def service_by_id(self) -> dict[str, dict]:
        return {s["service_id"]: s for s in self.services}

    @cached_property
    def customer_by_id(self) -> dict[str, dict]:
        return {c["customer_id"]: c for c in self.customers}

    @cached_property
    def post_by_id(self) -> dict[str, dict]:
        return {p["post_id"]: p for p in self.posts}

    @cached_property
    def category_name(self) -> dict[str, str]:
        return {c["category_id"]: c["name"] for c in self.categories}

    def experts_of(self, business_id: str) -> list[dict]:
        return [e for e in self.experts if e["business_id"] == business_id]

    def offerings_of(self, business_id: str) -> list[dict]:
        return [o for o in self.offerings if o["business_id"] == business_id]
