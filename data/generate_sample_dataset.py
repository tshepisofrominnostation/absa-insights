"""Generate the bundled sample dataset of SA e-commerce reviews (seeded).

SYNTHETIC DATA NOTICE: this generator produces realistic-but-synthetic
reviews so the pipeline is reproducible without scraping a live platform.
Swap in your own CSV (review_id,text) for real-world use — nothing in the
pipeline depends on this generator.

Seeded (SEED=42) so every run produces the identical 1,200-row CSV.
Usage: python data/generate_sample_dataset.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_REVIEWS = 1200

# (template, aspect) — sentiment-bearing clause pools. SA-flavoured.
POS = [
    ("delivery was quicker than I expected", "delivery"),
    ("the courier guy delivered a day early", "delivery"),
    ("parcel arrived in perfect condition", "delivery"),
    ("shipping was fast and the tracking worked", "delivery"),
    ("the price was really affordable", "price"),
    ("excellent value for money", "price"),
    ("definitely worth every rand", "price"),
    ("great quality, feels sturdy and well made", "quality"),
    ("the battery lasts me the whole day", "quality"),
    ("solid build, no complaints about the material", "quality"),
    ("customer service replied on WhatsApp within an hour", "customer service"),
    ("the support agent was friendly and helpful", "customer service"),
    ("exactly as described in the picture", "product accuracy"),
    ("the colour matches the photo perfectly", "product accuracy"),
    ("genuine product, not a replica", "product accuracy"),
    ("packaging was sealed and secure", "packaging"),
    ("loved the bubble wrap, box arrived undamaged", "packaging"),
    ("my refund was processed in two days", "returns & refunds"),
    ("the exchange was painless, got my replacement fast", "returns & refunds"),
]
NEG = [
    ("delivery was two weeks late", "delivery"),
    ("the courier never pitched and nobody called back", "delivery"),
    ("still waiting on my parcel after a month", "delivery"),
    ("shipping cost more than the item itself", "delivery"),
    ("the price is a rip-off for what you get", "price"),
    ("way overpriced for cheap material", "price"),
    ("broke after two days, really flimsy quality", "quality"),
    ("it stopped working within a week", "quality"),
    ("the battery is defective", "quality"),
    ("customer service never replied to my emails", "customer service"),
    ("the call centre kept me on hold for 40 minutes", "customer service"),
    ("support was useless, no response at all", "customer service"),
    ("not as described, the size is completely wrong", "product accuracy"),
    ("the colour looks nothing like the picture", "product accuracy"),
    ("received a fake, definitely a replica", "product accuracy"),
    ("packaging was damaged and the box was crushed", "packaging"),
    ("no bubble wrap, just a flimsy envelope", "packaging"),
    ("still no refund three weeks later", "returns & refunds"),
    ("they refused my return and ignored my messages", "returns & refunds"),
]
NEUTRAL = [
    ("delivery took about a week", "delivery"),
    ("the price is okay, nothing special", "price"),
    ("quality is average for this bracket", "quality"),
    ("support answered the next day with a standard reply", "customer service"),
    ("the size is more or less as described", "product accuracy"),
    ("packaging was a plain box", "packaging"),
    ("return process is still ongoing", "returns & refunds"),
]
OPENERS_POS = ["Great experience overall", "Lekker purchase", "Happy customer here",
               "Five stars from me", "Will order again", "Impressed start to finish"]
OPENERS_NEG = ["Disappointed to be honest", "Not impressed", "Eish, what a mission",
               "Will think twice before ordering again", "This was a headache"]
FILLER = ["ordered the mid-range option", "bought this for my mom",
          "second time buying from this store", "placed the order on payday",
          "it was during load shedding week so I was patient"]
PRODUCTS = ["kettle", "phone charger", "sneakers", "desk lamp", "backpack",
            "hair dryer", "Bluetooth speaker", "hoodie", "coffee machine",
            "power bank", "office chair", "yoga mat"]

def build_review(rng: random.Random) -> tuple[str, list[str]]:
    """Compose one review from clause pools; returns (text, aspects)."""
    roll = rng.random()
    if roll < 0.55:  # positive-dominant, as in most e-commerce data
        opener = rng.choice(OPENERS_POS)
        clauses = [opener]
        aspects = []
        for _ in range(rng.choice((1, 1, 2))):
            clause, aspect = rng.choice(POS)
            clauses.append(clause); aspects.append(aspect)
        if rng.random() < 0.18:  # occasional gripe
            clause, aspect = rng.choice(NEG)
            clauses.insert(rng.randint(1, len(clauses)), clause); aspects.append(aspect)
    elif roll < 0.75:
        opener = rng.choice(OPENERS_NEG)
        clauses = [opener]; aspects = []
        for _ in range(rng.choice((1, 2, 2))):
            clause, aspect = rng.choice(NEG)
            clauses.append(clause); aspects.append(aspect)
        if rng.random() < 0.25:
            clause, aspect = rng.choice(POS)
            clauses.append(clause); aspects.append(aspect)
    else:
        clauses, aspects = [], []
        for _ in range(rng.choice((1, 2))):
            if rng.random() < 0.35:
                clause, aspect = rng.choice(NEUTRAL)
                aspects.append(aspect)
            else:
                clause = rng.choice(FILLER)
            clauses.append(clause)
        if not aspects and rng.random() < 0.5:
            clause, aspect = rng.choice(NEUTRAL)
            clauses.append(clause); aspects.append(aspect)
    text = ", ".join(clauses[:-1]) + (". " if len(clauses) > 1 else "") + clauses[-1] + "."
    if rng.random() < 0.10:
        text = text + rng.choice([" 👍", " ⭐⭐⭐⭐⭐", " 😡", " 😍", " 🙂"])
    if rng.random() < 0.06:
        text = text.upper()
    return text, aspects


def main() -> None:
    """Write data/sample_reviews.csv with N_REVIEWS seeded rows."""
    rng = random.Random(SEED)
    rows = []
    start = date(2026, 7, 1)
    for i in range(N_REVIEWS):
        text, aspects = build_review(rng)
        n_pos = sum(1 for a in set(aspects))
        rating = 5
        rows.append({
            "review_id": f"s{i:04d}",
            "date": (start + timedelta(days=rng.randint(0, 60))).isoformat(),
            "product": rng.choice(PRODUCTS),
            "rating": rng.randint(1, 5),
            "text": text,
        })
    out = Path(__file__).parent / "sample_reviews.csv"
    pd_lines = ["review_id,date,product,rating,text"]
    for r in rows:
        t = r["text"].replace('"', "'")
        pd_lines.append(f'{r["review_id"]},{r["date"]},{r["product"]},'
                        f'{r["rating"]},"{t}"')
    out.write_text("\n".join(pd_lines) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
