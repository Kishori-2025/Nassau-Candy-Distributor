"""Static reference data for the Nassau Candy factory optimization app."""

from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "Nassau Candy Distributor.csv"

# Factory coordinates, taken from the project's Problem Overview ("Factories Co-ordinates").
FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

# Current product -> factory assignment.
# ASSUMPTION: the sales CSV has no factory column and the overview pages provided do not list
# this mapping. It follows the public Nassau Candy reference product table. Edit here if your
# copy of the brief says otherwise.
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory",
}

# Approximate geographic centroids (lat, lon) used as the customer location for each
# state/province, because the dataset only has city/state/postal code, not coordinates.
STATE_CENTROIDS = {
    "Alabama": (32.8, -86.8), "Arizona": (34.3, -111.7), "Arkansas": (34.9, -92.4),
    "California": (37.2, -119.4), "Colorado": (39.0, -105.5), "Connecticut": (41.6, -72.7),
    "Delaware": (39.0, -75.5), "District of Columbia": (38.9, -77.0), "Florida": (28.6, -82.4),
    "Georgia": (32.7, -83.4), "Idaho": (44.4, -114.6), "Illinois": (40.0, -89.2),
    "Indiana": (39.9, -86.3), "Iowa": (42.1, -93.5), "Kansas": (38.5, -98.4),
    "Kentucky": (37.5, -85.3), "Louisiana": (31.1, -92.0), "Maine": (45.4, -69.2),
    "Maryland": (39.0, -76.8), "Massachusetts": (42.3, -71.8), "Michigan": (44.3, -85.4),
    "Minnesota": (46.3, -94.3), "Mississippi": (32.7, -89.7), "Missouri": (38.4, -92.5),
    "Montana": (47.0, -109.6), "Nebraska": (41.5, -99.8), "Nevada": (39.3, -116.6),
    "New Hampshire": (43.7, -71.6), "New Jersey": (40.2, -74.7), "New Mexico": (34.4, -106.1),
    "New York": (42.9, -75.5), "North Carolina": (35.6, -79.4), "North Dakota": (47.5, -100.5),
    "Ohio": (40.3, -82.8), "Oklahoma": (35.6, -97.5), "Oregon": (43.9, -120.6),
    "Pennsylvania": (40.9, -77.8), "Rhode Island": (41.7, -71.5), "South Carolina": (33.9, -80.9),
    "South Dakota": (44.4, -100.2), "Tennessee": (35.9, -86.4), "Texas": (31.5, -99.3),
    "Utah": (39.3, -111.7), "Vermont": (44.1, -72.7), "Virginia": (37.5, -78.9),
    "Washington": (47.4, -120.5), "West Virginia": (38.6, -80.6), "Wisconsin": (44.6, -89.9),
    "Wyoming": (43.0, -107.6),
    # Canada
    "Alberta": (53.9, -116.6), "British Columbia": (53.7, -127.6), "Manitoba": (55.0, -97.0),
    "New Brunswick": (46.5, -66.2), "Newfoundland and Labrador": (53.1, -57.7),
    "Nova Scotia": (45.0, -63.0), "Ontario": (50.0, -85.0), "Prince Edward Island": (46.25, -63.0),
    "Quebec": (52.9, -73.5), "Saskatchewan": (52.9, -106.5),
}

RANDOM_STATE = 42
