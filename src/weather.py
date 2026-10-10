"""Current weather from Open-Meteo, turned into the five conditions the model knows."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from src.schema import WEATHER_SEVERITY

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 5
CACHE_SECONDS = 600

# Sustained wind in km/h: gale force and hurricane force on the Beaufort scale.
STORM_WIND_KMH = 62
HURRICANE_WIND_KMH = 118

FOG_CODES = {45, 48}
THUNDERSTORM_CODES = {95, 96, 99}

_cache = {}


def classify_weather(weather_code, wind_kmh):
    """Map a WMO weather code and wind speed to Clear, Rain, Fog, Storm or Hurricane."""
    wind = float(wind_kmh or 0)
    code = int(weather_code)

    if wind >= HURRICANE_WIND_KMH:
        return "Hurricane"
    if wind >= STORM_WIND_KMH or code in THUNDERSTORM_CODES:
        return "Storm"
    if code in FOG_CODES:
        return "Fog"
    if code >= 51:
        # Drizzle, rain, snow and showers all slow a shipment the way rain does.
        return "Rain"
    return "Clear"


def _fetch(points):
    query = urllib.parse.urlencode({
        "latitude": ",".join(f"{lat:.2f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.2f}" for _, lon in points),
        "current": "weather_code,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    })
    request = urllib.request.Request(
        f"{OPEN_METEO_URL}?{query}",
        headers={"User-Agent": "aura-dormathon/1.0"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    # One point comes back as an object, several as a list.
    return payload if isinstance(payload, list) else [payload]


def current_weather(points, fetch=None):
    """Worst current condition over the given (latitude, longitude) points.

    Returns (condition, note), or (None, None) when the service cannot be reached.
    """
    points = [tuple(point) for point in points]
    if not points:
        return None, None

    key = tuple(points)
    cached = _cache.get(key)
    if fetch is None and cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]

    try:
        readings = (fetch or _fetch)(points)
        conditions = [
            classify_weather(
                reading["current"]["weather_code"],
                reading["current"]["wind_speed_10m"],
            )
            for reading in readings
        ]
        if len(conditions) != len(points):
            return None, None
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        urllib.error.URLError,
    ):
        return None, None

    worst = max(conditions, key=WEATHER_SEVERITY.get)
    result = (
        worst,
        "Live reading from Open-Meteo: the worse of the current weather "
        "at the origin and the destination.",
    )
    if fetch is None:
        _cache[key] = (time.time(), result)
    return result
