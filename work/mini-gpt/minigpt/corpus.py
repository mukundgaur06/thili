"""Generates the default training corpus: short, simple children's stories.

Why synthetic? The workbench has no internet-dependent datasets and a CPU-only budget.
A 1M-parameter GPT trained for a few minutes cannot learn English from a few KB of real
prose, but it CAN learn the grammar, pronoun agreement and story structure of a
controlled language (the idea behind the TinyStories dataset, Eldan & Li, 2023).
The grammar below is original and seeded, so the corpus is reproducible.

Any plain-text file can be used instead: ``python train.py --corpus my.txt``.
Stories/documents are separated by blank lines.
"""

from __future__ import annotations

import random
from typing import List

HEROES = [
    # name, kind, subject pronoun, object pronoun, possessive
    ("Lily", "girl", "she", "her", "her"), ("Tom", "boy", "he", "him", "his"),
    ("Mia", "girl", "she", "her", "her"), ("Ben", "boy", "he", "him", "his"),
    ("Sara", "girl", "she", "her", "her"), ("Max", "boy", "he", "him", "his"),
    ("Anna", "girl", "she", "her", "her"), ("Leo", "boy", "he", "him", "his"),
    ("Pip", "little mouse", "he", "him", "his"), ("Rosie", "small rabbit", "she", "her", "her"),
    ("Bruno", "big brown dog", "he", "him", "his"), ("Luna", "curious cat", "she", "her", "her"),
    ("Ollie", "young owl", "he", "him", "his"), ("Daisy", "happy duck", "she", "her", "her"),
]
FRIENDS = [
    ("a kind old bear", "the bear"), ("a tiny bird", "the bird"), ("a friendly frog", "the frog"),
    ("a wise turtle", "the turtle"), ("a playful puppy", "the puppy"), ("a shy fox", "the fox"),
    ("a busy squirrel", "the squirrel"), ("a gentle horse", "the horse"),
]
PLACES = [
    "a small house near the forest", "a farm with a red barn", "a quiet village by the sea",
    "a big city full of lights", "a garden with many flowers", "a little hill under the sky",
    "a cozy home next to the river", "a park with tall green trees",
]
OBJECTS = [
    ("a red ball", "the ball"), ("a shiny key", "the key"), ("a blue kite", "the kite"),
    ("a wooden boat", "the boat"), ("a soft blanket", "the blanket"), ("a golden star", "the star"),
    ("a box of crayons", "the crayons"), ("a tiny bell", "the bell"), ("a big apple", "the apple"),
]
GOAL_PLACES = ["the top of the hill", "the other side of the river", "the old tree",
               "the edge of the pond", "the middle of the field", "the end of the path"]
FEELINGS_GOOD = ["happy", "proud", "excited", "glad", "thankful"]
FEELINGS_BAD = ["sad", "worried", "scared", "upset", "lonely"]
WEATHER = ["the sun was warm", "the wind was soft", "it started to rain", "the sky was grey",
           "snow fell slowly", "the stars were bright"]
TIMES = ["One day", "One morning", "One sunny afternoon", "Once upon a time", "One cold evening",
         "Long ago"]
LESSONS = [
    "It is good to share with friends.", "Asking for help is brave.",
    "Being kind makes everyone happy.", "Never give up when things are hard.",
    "Friends make every day better.", "Saying sorry can fix a lot.",
    "Patience helps you find what you need.",
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _story_lost(r: random.Random) -> List[str]:
    name, kind, he, him, his = r.choice(HEROES)
    obj, the_obj = r.choice(OBJECTS)
    fr, the_fr = r.choice(FRIENDS)
    place = r.choice(GOAL_PLACES)
    return [
        f"{r.choice(TIMES)}, a {kind} named {name} lived in {r.choice(PLACES)}.",
        f"{name} had {obj}. {_cap(he)} loved {the_obj} very much.",
        f"But one day, {the_obj} was gone. {name} felt {r.choice(FEELINGS_BAD)}.",
        f"{_cap(he)} looked under the bed and behind the door, but {he} could not find it.",
        f"Then {name} met {fr}. \"Can you help me?\" {he} asked.",
        f"{_cap(the_fr)} said, \"Yes! I saw {the_obj} near {place}.\"",
        f"They went to {place} together, and there was {the_obj}!",
        f"{name} was so {r.choice(FEELINGS_GOOD)}. {_cap(he)} gave {the_fr} a big hug.",
        r.choice(LESSONS),
    ]


def _story_share(r: random.Random) -> List[str]:
    a = r.choice(HEROES)
    b = r.choice([h for h in HEROES if h[0] != a[0]])
    obj, the_obj = r.choice(OBJECTS)
    return [
        f"{r.choice(TIMES)}, {a[0]} and {b[0]} were playing outside. {_cap(r.choice(WEATHER))}.",
        f"{a[0]} had {obj}, and {b[0]} wanted to play with it too.",
        f"\"No, it is mine!\" said {a[0]}. {b[0]} felt {r.choice(FEELINGS_BAD)} and walked away.",
        f"{a[0]} played alone, but it was not fun anymore.",
        f"{_cap(a[2])} ran to {b[0]} and said, \"I am sorry. Let us play together.\"",
        f"{b[0]} smiled. They played with {the_obj} all day long.",
        f"At night, {a[0]} and {b[0]} were very {r.choice(FEELINGS_GOOD)}.",
        r.choice(LESSONS),
    ]


def _story_journey(r: random.Random) -> List[str]:
    name, kind, he, him, his = r.choice(HEROES)
    place = r.choice(GOAL_PLACES)
    fr, the_fr = r.choice(FRIENDS)
    return [
        f"{r.choice(TIMES)}, there was a {kind} called {name}.",
        f"{name} wanted to go to {place}, but the way was long.",
        f"{_cap(he)} walked and walked. {_cap(r.choice(WEATHER))}, and {he} felt tired.",
        f"{_cap(he)} sat down and felt {r.choice(FEELINGS_BAD)}.",
        f"Then {fr} came by. \"Do not give up,\" said {the_fr}. \"I will walk with you.\"",
        f"Step by step, they reached {place}.",
        f"From there, {name} could see {his} home. {_cap(he)} felt {r.choice(FEELINGS_GOOD)}.",
        r.choice(LESSONS),
    ]


def _story_gift(r: random.Random) -> List[str]:
    name, kind, he, him, his = r.choice(HEROES)
    fr, the_fr = r.choice(FRIENDS)
    obj, the_obj = r.choice(OBJECTS)
    return [
        f"{r.choice(TIMES)}, {name} the {kind} wanted to make a gift for {his} mom.",
        f"{_cap(he)} found {obj} and painted it with bright colors.",
        f"But {he} dropped it, and it broke. {name} began to cry.",
        f"{_cap(fr)} heard {him} and came to help.",
        f"Together they fixed {the_obj}. It looked even better than before.",
        f"{name} gave {the_obj} to {his} mom. She smiled and said, \"I love it!\"",
        f"{name} felt {r.choice(FEELINGS_GOOD)}, and {he} thanked {the_fr}.",
        r.choice(LESSONS),
    ]


STORY_KINDS = [_story_lost, _story_share, _story_journey, _story_gift]


def make_stories(n: int, seed: int = 1337) -> List[str]:
    r = random.Random(seed)
    stories = []
    for _ in range(n):
        lines = r.choice(STORY_KINDS)(r)
        stories.append(" ".join(lines))
    return stories


def write_corpus(path, n: int = 3000, seed: int = 1337) -> int:
    stories = make_stories(n, seed)
    text = "\n\n".join(stories) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return len(text)
