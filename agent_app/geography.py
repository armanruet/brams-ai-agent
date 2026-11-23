"""Evidence-based, bounded geographic queries over BRAMS registry and archive."""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import math
from pathlib import Path

from .cancellation import check_cancelled
from .brams_sdk import (
    _catalog_from_sdk_with_meta,
    _fresh_sdk,
    _load_spectrogram,
    _locations_from_sdk_with_meta,
    _utc_string,
    filter_receivers,
)

BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "belgium-natural-earth-10m.geojson"


@lru_cache(maxsize=1)
def country_boundary():
    feature = json.loads(BOUNDARY_PATH.read_text())
    shape = feature["geometry"]
    polygons = [shape["coordinates"]] if shape["type"] == "Polygon" else shape["coordinates"]
    points = [point for polygon in polygons for ring in polygon for point in ring]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return polygons, (min(xs), min(ys), max(xs), max(ys)), feature["properties"]


def _inside_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10:
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def in_country(lon, lat, polygons):
    return any(_inside_ring(lon, lat, polygon[0]) and not any(_inside_ring(lon, lat, hole) for hole in polygon[1:]) for polygon in polygons)


def select_region(rows, locations, direction, boundary=None):
    polygons, (west, south, east, north), properties = country_boundary()
    axis = "latitude" if direction in {"north", "south"} else "longitude"
    mid = (south + north) / 2 if axis == "latitude" else (west + east) / 2
    cutoff = float(boundary) if boundary is not None else mid
    if not math.isfinite(cutoff) or not (south <= cutoff <= north if axis == "latitude" else west <= cutoff <= east):
        raise ValueError("The geographic boundary must lie within Belgium's extent.")
    grouped = {}
    missing = set()
    suspicious = set()
    for row in rows:
        code = row["location_code"]
        item = locations.get(code)
        if not item:
            missing.add(code)
            continue
        try:
            lat, lon = float(item["latitude"]), float(item["longitude"])
            if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            missing.add(code)
            continue
        if not in_country(lon, lat, polygons):
            if in_country(lat, lon, polygons):
                suspicious.add(code)
            continue
        value = lat if axis == "latitude" else lon
        if not (value >= cutoff if direction in {"north", "east"} else value < cutoff):
            continue
        site = grouped.setdefault(code, {"location_code": code, "name": item.get("name") or row["name"],
                                         "latitude": lat, "longitude": lon, "receivers": []})
        site["receivers"].append(row["receiver_id"])
    sites = sorted(grouped.values(), key=lambda site: (site["name"].casefold(), site["location_code"]))
    for site in sites:
        site["receivers"].sort()
    return {"sites": sites, "axis": axis, "cutoff_degrees": cutoff,
            "comparison": ">=" if direction in {"north", "east"} else "<",
            "definition_type": "user-specified" if boundary is not None else "half of Belgium's geographic extent",
            "missing_coordinate_sites": len(missing), "suspect_coordinate_sites": len(suspicious),
            "boundary_source_url": properties["source_url"], "boundary_source": properties["source"]}


def _recording_active_ids(entries, receivers, start, end):
    """Only count exact-window metadata with valid nominal and precise bounds."""
    allowed = set(receivers)
    found, uncertain = set(), set()
    for item in entries:
        rid = item.get("receiver_code")
        if rid not in allowed:
            continue
        try:
            nominal = datetime(*(int(item[k]) for k in ("year", "month", "day", "hours", "minutes")), tzinfo=timezone.utc)
            a = datetime.fromtimestamp(int(item["precise_start"]) / 1e6, timezone.utc)
            b = datetime.fromtimestamp(int(item["precise_end"]) / 1e6, timezone.utc)
            if b <= a:
                raise ValueError
            if nominal < end and nominal + timedelta(minutes=5) > start and a < end and b > start:
                found.add(rid)
        except (KeyError, ValueError, TypeError, OverflowError, OSError):
            uncertain.add(rid)
    return found, uncertain - found


def query_belgian_region(direction, action, start=None, end=None, boundary=None):
    from pybrams.brams.fetch import api as sdk_api
    with _fresh_sdk() as (Receiver, api, Config):
        catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
        rows = filter_receivers(catalog, "")
        endpoint = Config.get("pybrams.brams.location", "api_endpoint")
        raw, _ = _locations_from_sdk_with_meta(Receiver, api, endpoint, sdk_context=_fresh_sdk)
        if not isinstance(raw, list):
            raise RuntimeError("The location registry did not return a complete list.")
        locations = {item["location_code"]: item for item in raw if isinstance(item, dict) and item.get("location_code")}
        region = select_region(rows, locations, direction, boundary)
        sites = region.pop("sites")
        result = {**region, "status": "ok", "country": "Belgium", "direction": direction, "action": action,
                  "site_count": len(sites), "receiver_count": sum(len(site["receivers"]) for site in sites),
                  "sites": sites, "catalog_source_url": api.base_url + Config.get("pybrams.brams.receiver", "api_endpoint"),
                  "coordinates_source_url": api.base_url + endpoint, "recording_availability_checked": False}
        if action == "count":
            return result
        if not sites:
            return {**result, "status": "no_mappable_stations",
                    "message": "No stations could be confirmed in this region from usable coordinates; recording activity was not checked."}
        ids = [rid for site in sites for rid in site["receivers"]]
        file_endpoint = Config.get("pybrams.brams.file", "api_endpoint")
        payload = api.request(file_endpoint, {"from": start, "to": end, "receiver_code[]": ids})
        result["recording_source_url"] = api.base_url + file_endpoint
        result["start_utc"], result["end_utc"] = _utc_string(start), _utc_string(end)
        if payload is None:
            return {**result, "status": "availability_unknown", "message": "The archive returned no usable recording listing, so activity is unknown."}
        if not isinstance(payload, list):
            raise ValueError("Unexpected BRAMS recording listing")
        active_ids, uncertain_ids = _recording_active_ids(payload, ids, start, end)
        active_sites = [site for site in sites if set(site["receivers"]) & active_ids]
        unknown_sites = [site for site in sites if not set(site["receivers"]) & active_ids and set(site["receivers"]) & uncertain_ids]
        result.update(recording_availability_checked=True, active_site_count=len(active_sites),
                      active_receiver_count=len(active_ids), unknown_activity_site_count=len(unknown_sites),
                      active_sites=active_sites, active_receiver_ids=sorted(active_ids))
    if action == "active_count":
        return result
    if not active_ids:
        return {**result, "status": "no_recordings_found" if not unknown_sites else "availability_unknown",
                "message": "No matching recording was confirmed for this interval."}
    # The SDK download occurs outside the registry lock; select only confirmed receivers.
    for rid in sorted(active_ids):
        check_cancelled()
        try:
            image = _load_spectrogram(rid, start, end)
        except Exception:
            continue
        if image.get("status") == "ok" and image.get("downloaded"):
            try:
                image_start = datetime.fromisoformat(image["recording_start_utc"].replace("Z", "+00:00"))
                nominal = image_start.replace(minute=(image_start.minute // 5) * 5, second=0, microsecond=0)
                if nominal >= end or nominal + timedelta(minutes=5) <= start:
                    continue
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
            return {**result, **image, "action": action}
    return {**result, "status": "spectrogram_unavailable", "message": "Recordings were listed, but an archived spectrogram could not be retrieved."}
