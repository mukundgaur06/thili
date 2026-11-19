"""A 1,000-product catalogue for a synthetic hypermarket.

Products are SKUs: a *base product* ("Wireless Mouse") sold by an *invented brand* in a
*variant* (size, capacity, edition), with a price. Brands are invented on purpose: a
pretrained language model has read about real brands, and that prior would blur what it
learns from the baskets. Base products keep their ordinary English names, so the
off-the-shelf model can use what it knows about "bread" and "butter"; whether that helps is
one of the things measured.

Popularity flows down four levels, each normalised on its own:

    category weight -> base product within category -> brand within base -> variant within brand
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

# category: (weight, brands {name: weight}, [(base, weight, {variant: (weight, price multiplier)}), ...])
# Prices are in an invented currency unit; the multiplier scales the base price below.
CATALOGUE_SPEC: Dict[str, dict] = {
    "Bakery": {"weight": 9, "brands": {"Oven Lane": 5, "Golden Crust": 3, "Millstone": 2, "Rye & Co": 1},
               "products": [("Bread", 20, 2.5), ("Bagels", 5, 3.5), ("Croissants", 5, 4.0), ("Tortillas", 5, 3.0),
                            ("Muffins", 4, 4.5), ("Burger Buns", 5, 3.0)]},
    "Dairy": {"weight": 12, "brands": {"Meadow Farm": 5, "Cloud Nine Dairy": 3, "Alpine Pure": 2, "Bluebell Creamery": 1},
              "products": [("Milk", 25, 1.5), ("Butter", 10, 3.0), ("Cheddar Cheese", 8, 5.0), ("Yogurt", 9, 1.2),
                           ("Eggs", 14, 3.5), ("Cream Cheese", 5, 2.8), ("Mozzarella", 5, 4.0)]},
    "Beverages": {"weight": 10, "brands": {"Brewhaus": 4, "Morning Ritual": 3, "Fizzco": 3, "Tidewater": 1},
                  "products": [("Coffee", 12, 8.0), ("Tea", 8, 4.0), ("Orange Juice", 8, 3.5), ("Sparkling Water", 7, 1.0),
                               ("Cola", 10, 1.5), ("Energy Drink", 4, 2.5)]},
    "Pantry": {"weight": 11, "brands": {"Casa Verde": 4, "Harvest Table": 3, "Spice Route": 2, "Pantrywise": 1},
               "products": [("Pasta", 12, 2.0), ("Pasta Sauce", 8, 3.0), ("Rice", 10, 4.0), ("Olive Oil", 6, 9.0),
                            ("Cereal", 9, 4.5), ("Peanut Butter", 5, 3.5), ("Jam", 5, 3.0), ("Salsa", 4, 3.0),
                            ("Canned Tomatoes", 5, 1.5), ("Flour", 4, 2.0), ("Sugar", 6, 2.0)]},
    "Snacks": {"weight": 8, "brands": {"Crunchworks": 4, "Happy Bite": 3, "Nutty Path": 2, "Snaxly": 1},
               "products": [("Potato Chips", 10, 2.5), ("Chocolate Bar", 9, 1.5), ("Cookies", 7, 3.0),
                            ("Popcorn", 4, 2.5), ("Trail Mix", 3, 5.0), ("Pretzels", 3, 2.5)]},
    "Produce": {"weight": 12, "brands": {"": 1},
                "products": [("Bananas", 18, 1.5), ("Apples", 14, 3.0), ("Tomatoes", 12, 2.5), ("Onions", 12, 1.5),
                             ("Avocados", 6, 4.0), ("Lemons", 5, 2.0), ("Spinach", 5, 2.5), ("Garlic", 6, 1.0),
                             ("Limes", 4, 2.0), ("Potatoes", 10, 3.0)]},
    "Electronics": {"weight": 7, "brands": {"Voltix": 4, "Nimbus Tech": 3, "Arcwave": 2, "Kepler Labs": 1},
                    "products": [("Smartphone", 4, 400.0), ("Phone Case", 6, 15.0), ("Laptop", 2, 800.0),
                                 ("Wireless Mouse", 4, 25.0), ("Laptop Bag", 2, 40.0), ("USB Cable", 7, 8.0),
                                 ("Headphones", 4, 60.0), ("Power Bank", 4, 30.0), ("Screen Protector", 4, 10.0),
                                 ("Keyboard", 2, 45.0), ("AA Batteries", 8, 6.0)]},
    "Home Appliances": {"weight": 4, "brands": {"Hearthline": 3, "Kitchora": 3, "Brightwell": 2, "Homestead": 1},
                        "products": [("Coffee Maker", 3, 70.0), ("Coffee Filters", 6, 4.0), ("Toaster", 3, 35.0),
                                     ("Blender", 3, 50.0), ("Kettle", 3, 30.0), ("Light Bulbs", 8, 8.0),
                                     ("Vacuum Bags", 3, 10.0), ("Air Fryer", 2, 90.0)]},
    "Beauty": {"weight": 7, "brands": {"Lumiere": 4, "Pure Petal": 3, "Northglow": 2, "Velvetine": 1},
               "products": [("Shampoo", 10, 6.0), ("Conditioner", 7, 6.0), ("Body Wash", 7, 5.0),
                            ("Toothpaste", 10, 3.0), ("Toothbrush", 6, 3.5), ("Deodorant", 6, 4.5),
                            ("Face Cream", 4, 12.0), ("Sunscreen", 4, 10.0), ("Razor", 4, 8.0)]},
    "Sports": {"weight": 4, "brands": {"Stridewell": 3, "Peakform": 3, "Coreline": 2, "Trailmark": 1},
               "products": [("Running Shoes", 3, 90.0), ("Sports Socks", 6, 8.0), ("Water Bottle", 6, 12.0),
                            ("Yoga Mat", 3, 25.0), ("Protein Bar", 7, 2.5), ("Tennis Balls", 3, 6.0),
                            ("Gym Towel", 3, 10.0)]},
    "Fashion": {"weight": 4, "brands": {"Threadline": 3, "Urban Loom": 3, "Kestrel": 2, "Northfield": 1},
                "products": [("T-Shirt", 6, 15.0), ("Jeans", 4, 45.0), ("Belt", 3, 20.0), ("Socks", 7, 6.0),
                             ("Baseball Cap", 3, 18.0), ("Rain Jacket", 2, 70.0), ("Umbrella", 3, 15.0)]},
    "Books": {"weight": 3, "brands": {"Quill House": 3, "Lantern Press": 3, "Bluepine Books": 2, "Paper Moon": 1},
              "products": [("Novel", 8, 14.0), ("Cookbook", 4, 25.0), ("Notebook", 7, 4.0), ("Bookmark", 3, 2.0),
                           ("Children's Book", 4, 10.0), ("Puzzle Book", 3, 8.0), ("Pens", 6, 5.0)]},
}

# Variants (and their within-brand weight and price multiplier), shared per category.
VARIANTS = {
    "Bakery": {"Classic": (5, 1.0), "Whole Wheat": (3, 1.15), "Family Pack": (2, 1.8)},
    "Dairy": {"Regular": (6, 1.0), "Organic": (2, 1.5), "Large": (3, 1.7)},
    "Beverages": {"Regular": (6, 1.0), "Large": (3, 1.8), "Premium": (2, 1.6)},
    "Pantry": {"Regular": (6, 1.0), "Organic": (2, 1.4), "Value Size": (3, 1.7)},
    "Snacks": {"Regular": (6, 1.0), "Share Size": (3, 1.8), "Mini": (2, 0.6)},
    "Produce": {"Loose": (5, 1.0), "Pack": (4, 1.4), "Organic": (2, 1.6)},
    "Electronics": {"Standard": (6, 1.0), "Pro": (2, 1.6), "Lite": (3, 0.75)},
    "Home Appliances": {"Standard": (6, 1.0), "Deluxe": (2, 1.5), "Compact": (3, 0.8)},
    "Beauty": {"Regular": (6, 1.0), "Travel Size": (3, 0.5), "Salon": (2, 1.7)},
    "Sports": {"Standard": (6, 1.0), "Pro": (2, 1.7), "Kids": (2, 0.7)},
    "Fashion": {"Standard": (6, 1.0), "Premium": (2, 1.8), "Kids": (2, 0.7)},
    "Books": {"Paperback": (6, 1.0), "Hardcover": (3, 1.8), "Pocket": (2, 0.6)},
}


@dataclass(frozen=True)
class Product:
    pid: int          # 0 .. n-1, stable; the language-model token is <p{pid:04d}>
    sku: str          # "P0042"
    name: str         # "Meadow Farm Milk Organic"
    base: str         # "Milk"
    category: str     # "Dairy"
    brand: str        # "Meadow Farm" ("" for loose produce)
    variant: str      # "Organic"
    price: float
    tier: int         # 0 budget, 1 standard, 2 premium, by price rank within the base product

    def to_dict(self) -> dict:
        return asdict(self)


def _all_skus() -> List[dict]:
    rows = []
    for cat, spec in CATALOGUE_SPEC.items():
        for base, w_base, price in spec["products"]:
            for brand, w_brand in spec["brands"].items():
                for variant, (w_var, mult) in VARIANTS[cat].items():
                    name = " ".join(p for p in (brand, base, variant) if p)
                    rows.append({"category": cat, "base": base, "brand": brand, "variant": variant, "name": name,
                                 "price": round(price * mult * (1 + 0.08 * (len(brand) % 3)), 2),
                                 "w": (spec["weight"], w_base, w_brand, w_var)})
    return rows


def make_catalogue(n_products: int = 1000, seed: int = 0) -> List[Product]:
    """Exactly `n_products` SKUs, every base product represented, chosen with a fixed seed."""
    pool = _all_skus()
    rng = np.random.default_rng(seed)
    by_base: Dict[str, List[int]] = {}
    for i, r in enumerate(pool):
        by_base.setdefault(r["base"], []).append(i)
    if n_products < len(by_base):
        raise ValueError(f"need at least {len(by_base)} products to cover every base product")
    if n_products > len(pool):
        raise ValueError(f"only {len(pool)} SKUs can be generated")
    chosen = {int(rng.choice(ix)) for ix in by_base.values()}
    rest = np.array([i for i in range(len(pool)) if i not in chosen])
    chosen |= set(int(i) for i in rng.choice(rest, n_products - len(chosen), replace=False))
    rows = [pool[i] for i in sorted(chosen)]
    tiers = {}
    for base in by_base:
        prices = sorted({r["price"] for r in rows if r["base"] == base})
        for r in rows:
            if r["base"] == base:
                k = prices.index(r["price"])
                tiers[r["name"]] = 0 if len(prices) == 1 else min(2, int(3 * k / len(prices)))
    return [Product(pid=i, sku=f"P{i:04d}", name=r["name"], base=r["base"], category=r["category"], brand=r["brand"],
                    variant=r["variant"], price=r["price"], tier=tiers[r["name"]]) for i, r in enumerate(rows)]


def popularity(catalogue: List[Product]) -> np.ndarray:
    """Each product's share of sales (sums to 1). Deterministic: no randomness."""
    w_cat = {c: s["weight"] for c, s in CATALOGUE_SPEC.items()}
    w_base = {(c, b): w for c, s in CATALOGUE_SPEC.items() for b, w, _ in s["products"]}
    w_brand = {(c, br): w for c, s in CATALOGUE_SPEC.items() for br, w in s["brands"].items()}
    stocked_bases, stocked_brands, stocked_vars = {}, {}, {}
    for p in catalogue:
        stocked_bases.setdefault(p.category, set()).add(p.base)
        stocked_brands.setdefault(p.base, set()).add(p.brand)
        stocked_vars.setdefault((p.base, p.brand), set()).add(p.variant)
    tot_cat = sum(w_cat.values())
    shares = []
    for p in catalogue:
        s = w_cat[p.category] / tot_cat
        s *= w_base[(p.category, p.base)] / sum(w_base[(p.category, b)] for b in stocked_bases[p.category])
        s *= w_brand[(p.category, p.brand)] / sum(w_brand[(p.category, b)] for b in stocked_brands[p.base])
        vw = VARIANTS[p.category]
        s *= vw[p.variant][0] / sum(vw[v][0] for v in stocked_vars[(p.base, p.brand)])
        shares.append(s)
    shares = np.array(shares)
    return shares / shares.sum()


def find(catalogue: List[Product], base: str) -> List[Product]:
    return [p for p in catalogue if p.base == base]
