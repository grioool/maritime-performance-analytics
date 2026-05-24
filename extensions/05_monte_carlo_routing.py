# %%
from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "MasterSet_features.csv"
RESULTS_DIR = ROOT / "results"
LAND_GEOJSON_PATH = ROOT / "data" / "geo" / "ne_10m_land.geojson"

KN_TO_MS = 0.51444
NM_PER_DEG_LAT = 60.0
P_DESIGN_KW = 3500.0
V_DESIGN_KN = 11.5
SFOC_G_KWH = 195.0
EF_HFO = 3.114
KW_PER_KN = 200.0

WIND_SPEEDS = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], dtype=float)
WIND_ANGLES = np.array([0, 20, 40, 60, 80, 90, 100, 120, 140, 160, 180], dtype=float)
POWER_TABLE = np.array(
    [
        [0, 3, 10, 14, 16, 14, 12, 7, 3, 1, 0],
        [-2, 6, 22, 38, 44, 42, 38, 22, 9, 3, 0],
        [-5, 12, 45, 78, 90, 87, 78, 46, 19, 7, 0],
        [-8, 20, 75, 128, 148, 144, 130, 76, 32, 11, 0],
        [-12, 30, 110, 186, 215, 210, 188, 110, 46, 16, 0],
        [-16, 42, 152, 256, 297, 290, 260, 152, 64, 22, 0],
        [-21, 56, 200, 335, 390, 380, 340, 199, 83, 29, 0],
        [-27, 72, 254, 425, 496, 484, 433, 254, 106, 37, 0],
        [-33, 90, 315, 526, 614, 599, 537, 314, 131, 46, 0],
        [-40, 110, 382, 638, 745, 727, 651, 381, 159, 55, 0],
        [-48, 132, 456, 762, 890, 868, 778, 455, 190, 66, 0],
    ],
    dtype=float,
)

ROTOR_INTERP = RegularGridInterpolator(
    (WIND_SPEEDS, WIND_ANGLES),
    POWER_TABLE,
    method="linear",
    bounds_error=False,
    fill_value=None,
)


@dataclass(frozen=True)
class Config:
    samples: int = 250
    seed: int = 42
    voyage_id: int | None = None
    grid_step_deg: float = 0.35
    weather_cell_deg: float = 0.7
    corridor_margin_deg: float = 0.9
    observed_water_radius_nm: float = 55.0
    max_water_radius_nm: float = 125.0
    base_speed_kn: float | None = None
    min_speed_kn: float = 5.0
    max_speed_kn: float = 18.0
    wave_calm_m: float = 1.0
    wave_loss_kn_per_m: float = 0.35
    risk_lambda: float = 0.35
    stop_gap_hours: float = 2.0
    land_geojson_path: Path = LAND_GEOJSON_PATH
    disable_land_mask: bool = False
    land_override_radius_nm: float = 14.0
    scenario_plot_count: int = 30


@dataclass
class Graph:
    grid_lats: np.ndarray
    grid_lons: np.ndarray
    allowed: np.ndarray
    node_to_ij: list[tuple[int, int]]
    ij_to_node: dict[tuple[int, int], int]
    edges: pd.DataFrame
    adjacency: list[list[int]]
    edge_lookup: dict[tuple[int, int], int]
    source: int
    target: int
    observed_radius_nm: float
    land_mask_enabled: bool


@dataclass(frozen=True)
class Leg:
    leg_id: int
    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    observed_distance_nm: float
    straight_distance_nm: float
    n_points: int
    source: int | None = None
    target: int | None = None


@dataclass
class ScenarioMetrics:
    fuel_base_t: np.ndarray
    fuel_rotor_t: np.ndarray
    hours_no_rotor: np.ndarray
    hours_speed_assist: np.ndarray
    rotor_energy_kwh: np.ndarray
    distance_nm: np.ndarray


@dataclass
class LandRing:
    lon: np.ndarray
    lat: np.ndarray
    bbox: tuple[float, float, float, float]


@dataclass
class LandPolygon:
    outer: LandRing
    holes: list[LandRing]


class LandMask:
    """Minimal GeoJSON land-mask geometry for the route graph."""

    def __init__(self, polygons: list[LandPolygon]):
        self.polygons = polygons
        seg_lon1: list[np.ndarray] = []
        seg_lat1: list[np.ndarray] = []
        seg_lon2: list[np.ndarray] = []
        seg_lat2: list[np.ndarray] = []

        for polygon in polygons:
            lon = polygon.outer.lon
            lat = polygon.outer.lat
            if len(lon) < 2:
                continue
            seg_lon1.append(lon[:-1])
            seg_lat1.append(lat[:-1])
            seg_lon2.append(lon[1:])
            seg_lat2.append(lat[1:])

        if seg_lon1:
            self.seg_lon1 = np.concatenate(seg_lon1)
            self.seg_lat1 = np.concatenate(seg_lat1)
            self.seg_lon2 = np.concatenate(seg_lon2)
            self.seg_lat2 = np.concatenate(seg_lat2)
            self.seg_min_lon = np.minimum(self.seg_lon1, self.seg_lon2)
            self.seg_max_lon = np.maximum(self.seg_lon1, self.seg_lon2)
            self.seg_min_lat = np.minimum(self.seg_lat1, self.seg_lat2)
            self.seg_max_lat = np.maximum(self.seg_lat1, self.seg_lat2)
        else:
            empty = np.array([], dtype=float)
            self.seg_lon1 = empty
            self.seg_lat1 = empty
            self.seg_lon2 = empty
            self.seg_lat2 = empty
            self.seg_min_lon = empty
            self.seg_max_lon = empty
            self.seg_min_lat = empty
            self.seg_max_lat = empty

    @classmethod
    def from_geojson(
        cls,
        path: Path,
        bounds: tuple[float, float, float, float],
        margin_deg: float = 0.3,
    ) -> "LandMask":
        if not path.exists():
            raise FileNotFoundError(
                f"Land GeoJSON not found at {path}. Download Natural Earth "
                "ne_10m_land.geojson or pass --disable-land-mask."
            )

        min_lon, min_lat, max_lon, max_lat = bounds
        query_bbox = (
            min_lon - margin_deg,
            min_lat - margin_deg,
            max_lon + margin_deg,
            max_lat + margin_deg,
        )

        with path.open("r", encoding="utf-8") as f:
            geo = json.load(f)

        polygons: list[LandPolygon] = []
        for feature in geo.get("features", []):
            geometry = feature.get("geometry") or {}
            geom_type = geometry.get("type")
            coords = geometry.get("coordinates", [])
            if geom_type == "Polygon":
                polygon_coords = [coords]
            elif geom_type == "MultiPolygon":
                polygon_coords = coords
            else:
                continue

            for rings in polygon_coords:
                if not rings:
                    continue
                outer = make_ring(rings[0])
                if outer is None or not bbox_overlaps(outer.bbox, query_bbox):
                    continue
                holes = [ring for ring_coords in rings[1:] if (ring := make_ring(ring_coords)) is not None]
                polygons.append(LandPolygon(outer=outer, holes=holes))

        return cls(polygons)

    def is_land(self, lon: float, lat: float) -> bool:
        for polygon in self.polygons:
            if not point_in_bbox(lon, lat, polygon.outer.bbox):
                continue
            if not point_in_ring(lon, lat, polygon.outer):
                continue
            if any(point_in_bbox(lon, lat, hole.bbox) and point_in_ring(lon, lat, hole) for hole in polygon.holes):
                return False
            return True
        return False

    def segment_crosses_land(self, lon1: float, lat1: float, lon2: float, lat2: float) -> bool:
        mid_lon = (lon1 + lon2) / 2.0
        mid_lat = (lat1 + lat2) / 2.0
        if self.is_land(lon1, lat1) or self.is_land(mid_lon, mid_lat) or self.is_land(lon2, lat2):
            return True
        return self.segment_intersects_coast(lon1, lat1, lon2, lat2)

    def segment_intersects_coast(self, lon1: float, lat1: float, lon2: float, lat2: float) -> bool:
        if len(self.seg_lon1) == 0:
            return False

        min_lon, max_lon = sorted((lon1, lon2))
        min_lat, max_lat = sorted((lat1, lat2))
        candidates = (
            (self.seg_max_lon >= min_lon)
            & (self.seg_min_lon <= max_lon)
            & (self.seg_max_lat >= min_lat)
            & (self.seg_min_lat <= max_lat)
        )
        if not np.any(candidates):
            return False

        qx1 = self.seg_lon1[candidates]
        qy1 = self.seg_lat1[candidates]
        qx2 = self.seg_lon2[candidates]
        qy2 = self.seg_lat2[candidates]

        rx = lon2 - lon1
        ry = lat2 - lat1
        sx = qx2 - qx1
        sy = qy2 - qy1
        denom = cross2(rx, ry, sx, sy)
        qpx = qx1 - lon1
        qpy = qy1 - lat1

        non_parallel = np.abs(denom) > 1e-12
        if np.any(non_parallel):
            t = cross2(qpx[non_parallel], qpy[non_parallel], sx[non_parallel], sy[non_parallel]) / denom[non_parallel]
            u = cross2(qpx[non_parallel], qpy[non_parallel], rx, ry) / denom[non_parallel]
            if np.any((t >= -1e-10) & (t <= 1.0 + 1e-10) & (u >= -1e-10) & (u <= 1.0 + 1e-10)):
                return True

        parallel = ~non_parallel
        if np.any(parallel):
            collinear = np.abs(cross2(qpx[parallel], qpy[parallel], rx, ry)) <= 1e-12
            if np.any(collinear):
                return True

        return False


def cross2(ax: np.ndarray | float, ay: np.ndarray | float, bx: np.ndarray | float, by: np.ndarray | float) -> np.ndarray:
    return np.asarray(ax) * np.asarray(by) - np.asarray(ay) * np.asarray(bx)


def make_ring(coords: list[list[float]]) -> LandRing | None:
    if len(coords) < 4:
        return None
    arr = np.asarray(coords, dtype=float)
    lon = arr[:, 0]
    lat = arr[:, 1]
    return LandRing(
        lon=lon,
        lat=lat,
        bbox=(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())),
    )


def bbox_overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def point_in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def point_in_ring(lon: float, lat: float, ring: LandRing) -> bool:
    xs = ring.lon
    ys = ring.lat
    xj = np.roll(xs, 1)
    yj = np.roll(ys, 1)
    crosses = ((ys > lat) != (yj > lat)) & (
        lon < (xj - xs) * (lat - ys) / np.where(np.abs(yj - ys) < 1e-12, 1e-12, yj - ys) + xs
    )
    return bool(np.count_nonzero(crosses) % 2 == 1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=Config.samples)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--voyage-id", type=int, default=None)
    parser.add_argument("--grid-step-deg", type=float, default=Config.grid_step_deg)
    parser.add_argument("--weather-cell-deg", type=float, default=Config.weather_cell_deg)
    parser.add_argument("--observed-water-radius-nm", type=float, default=Config.observed_water_radius_nm)
    parser.add_argument("--risk-lambda", type=float, default=Config.risk_lambda)
    parser.add_argument("--stop-gap-hours", type=float, default=Config.stop_gap_hours)
    parser.add_argument("--land-geojson", type=Path, default=LAND_GEOJSON_PATH)
    parser.add_argument("--disable-land-mask", action="store_true")
    parser.add_argument("--land-override-radius-nm", type=float, default=Config.land_override_radius_nm)
    parser.add_argument("--scenario-plot-count", type=int, default=Config.scenario_plot_count)
    args = parser.parse_args()
    return Config(
        samples=args.samples,
        seed=args.seed,
        voyage_id=args.voyage_id,
        grid_step_deg=args.grid_step_deg,
        weather_cell_deg=args.weather_cell_deg,
        observed_water_radius_nm=args.observed_water_radius_nm,
        risk_lambda=args.risk_lambda,
        stop_gap_hours=args.stop_gap_hours,
        land_geojson_path=args.land_geojson,
        disable_land_mask=args.disable_land_mask,
        land_override_radius_nm=args.land_override_radius_nm,
        scenario_plot_count=args.scenario_plot_count,
    )


def normalize_angle(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(angle_deg) + 180.0) % 360.0) - 180.0


def rotor_power_kw(wind_ms: np.ndarray, alpha_abs: np.ndarray) -> np.ndarray:
    ws = np.clip(np.asarray(wind_ms, dtype=float), WIND_SPEEDS.min(), WIND_SPEEDS.max())
    wa = np.clip(np.asarray(alpha_abs, dtype=float), WIND_ANGLES.min(), WIND_ANGLES.max())
    pts = np.column_stack([ws.ravel(), wa.ravel()])
    return ROTOR_INTERP(pts).reshape(ws.shape)


def brake_power_kw(sog_kn: np.ndarray | float) -> np.ndarray:
    v_ms = np.asarray(sog_kn, dtype=float) * KN_TO_MS
    k_resistance = P_DESIGN_KW / ((V_DESIGN_KN * KN_TO_MS) ** 3)
    return k_resistance * v_ms**3


def fuel_rate_t_h(power_kw: np.ndarray | float) -> np.ndarray:
    return SFOC_G_KWH * np.maximum(power_kw, 0.0) / 1e6


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_nm * 2 * math.asin(math.sqrt(a))


def course_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.degrees(math.atan2(dlon, dlat)) % 360.0


def project_nm(lat: np.ndarray, lon: np.ndarray, ref_lat: float) -> np.ndarray:
    x = lon * NM_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    y = lat * NM_PER_DEG_LAT
    return np.column_stack([x, y])


def load_data(path: Path) -> pd.DataFrame:
    cols = [
        "Time",
        "Lat",
        "Lon",
        "Voyage_ID",
        "Speed_kn",
        "Course_deg",
        "Dist_since_last_nm",
        "True_Wind_Speed_ms",
        "wind_dir_deg",
        "H_s",
        "uo",
        "vo",
    ]
    df = pd.read_csv(path, usecols=cols, parse_dates=["Time"])
    df = df.dropna(subset=cols).sort_values("Time").reset_index(drop=True)
    df["month"] = df["Time"].dt.month
    return df


def select_reference_voyage(df: pd.DataFrame, voyage_id: int | None) -> int:
    if voyage_id is not None:
        if voyage_id not in set(df["Voyage_ID"]):
            raise ValueError(f"Voyage_ID {voyage_id} is not present in the data.")
        return int(voyage_id)

    voyages = (
        df.groupby("Voyage_ID")
        .agg(
            n=("Speed_kn", "count"),
            start=("Time", "min"),
            lat_min=("Lat", "min"),
            lat_max=("Lat", "max"),
            lon_min=("Lon", "min"),
            lon_max=("Lon", "max"),
        )
        .reset_index()
    )
    voyages["lat_span"] = voyages["lat_max"] - voyages["lat_min"]
    voyages["lon_span"] = voyages["lon_max"] - voyages["lon_min"]
    voyages["span_score"] = voyages["lat_span"] + 0.35 * voyages["lon_span"]
    candidates = voyages[(voyages["n"] >= 150) & (voyages["span_score"] >= 2.5)].copy()
    if candidates.empty:
        candidates = voyages[voyages["n"] >= 50].copy()
    return int(candidates.sort_values(["span_score", "n"], ascending=False).iloc[0]["Voyage_ID"])


def detect_itinerary_legs(voyage: pd.DataFrame, stop_gap_hours: float) -> tuple[list[Leg], pd.DataFrame]:
    """Split a voyage into sailing legs separated by likely port/stop gaps.

    The engineered feature file mostly contains underway observations, so stops
    appear as unusually long time gaps between two nearby AIS points rather than
    as rows with zero speed. We preserve these gaps as forced itinerary breaks.
    """

    v = voyage.reset_index(drop=True).copy()
    gap_hours = v["Time"].diff().dt.total_seconds().div(3600).fillna(0.0)
    split_starts = np.flatnonzero(gap_hours.to_numpy() > stop_gap_hours).tolist()
    cuts = [0] + split_starts + [len(v)]

    legs: list[Leg] = []
    for leg_id, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        end = stop - 1
        if end <= start:
            continue
        segment = v.iloc[start : end + 1]
        start_row = segment.iloc[0]
        end_row = segment.iloc[-1]
        observed_dist = float(segment["Dist_since_last_nm"].iloc[1:].clip(lower=0.0).sum())
        straight_dist = haversine_nm(
            float(start_row["Lat"]),
            float(start_row["Lon"]),
            float(end_row["Lat"]),
            float(end_row["Lon"]),
        )
        legs.append(
            Leg(
                leg_id=leg_id,
                start_idx=int(start),
                end_idx=int(end),
                start_time=start_row["Time"],
                end_time=end_row["Time"],
                start_lat=float(start_row["Lat"]),
                start_lon=float(start_row["Lon"]),
                end_lat=float(end_row["Lat"]),
                end_lon=float(end_row["Lon"]),
                observed_distance_nm=observed_dist,
                straight_distance_nm=straight_dist,
                n_points=int(len(segment)),
            )
        )

    stop_rows = []
    for stop_id, departure_idx in enumerate(split_starts, start=1):
        arrival_idx = departure_idx - 1
        arrival = v.iloc[arrival_idx]
        departure = v.iloc[departure_idx]
        stop_rows.append(
            {
                "stop_id": stop_id,
                "arrival_idx": int(arrival_idx),
                "departure_idx": int(departure_idx),
                "arrival_time": arrival["Time"],
                "departure_time": departure["Time"],
                "gap_hours": float(gap_hours.iloc[departure_idx]),
                "arrival_lat": float(arrival["Lat"]),
                "arrival_lon": float(arrival["Lon"]),
                "departure_lat": float(departure["Lat"]),
                "departure_lon": float(departure["Lon"]),
                "stop_lat": float((arrival["Lat"] + departure["Lat"]) / 2.0),
                "stop_lon": float((arrival["Lon"] + departure["Lon"]) / 2.0),
                "gap_distance_nm": haversine_nm(
                    float(arrival["Lat"]),
                    float(arrival["Lon"]),
                    float(departure["Lat"]),
                    float(departure["Lon"]),
                ),
            }
        )

    return legs, pd.DataFrame(stop_rows)


def month_window(month: int) -> set[int]:
    return {((month - 2) % 12) + 1, month, (month % 12) + 1}


def grid_centres(min_val: float, max_val: float, step: float) -> np.ndarray:
    start = math.floor(min_val / step) * step
    stop = math.ceil(max_val / step) * step
    return np.round(np.arange(start, stop + step * 0.5, step), 8)


def route_bounds(voyage: pd.DataFrame, margin_deg: float) -> tuple[float, float, float, float]:
    return (
        float(voyage["Lon"].min() - margin_deg),
        float(voyage["Lat"].min() - margin_deg),
        float(voyage["Lon"].max() + margin_deg),
        float(voyage["Lat"].max() + margin_deg),
    )


def build_observed_water_tree(df: pd.DataFrame, ref_lat: float) -> cKDTree:
    obs_xy = project_nm(df["Lat"].to_numpy(), df["Lon"].to_numpy(), ref_lat)
    return cKDTree(obs_xy)


def point_allowed(tree: cKDTree, lat: float, lon: float, ref_lat: float, radius_nm: float) -> bool:
    xy = project_nm(np.array([lat]), np.array([lon]), ref_lat)
    distance, _ = tree.query(xy, k=1)
    return bool(distance[0] <= radius_nm)


def edge_near_track(
    tree: cKDTree,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    ref_lat: float,
    radius_nm: float,
) -> bool:
    samples = np.array(
        [
            (lat1, lon1),
            ((lat1 * 3 + lat2) / 4.0, (lon1 * 3 + lon2) / 4.0),
            ((lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0),
            ((lat1 + lat2 * 3) / 4.0, (lon1 + lon2 * 3) / 4.0),
            (lat2, lon2),
        ]
    )
    distances, _ = tree.query(project_nm(samples[:, 0], samples[:, 1], ref_lat), k=1)
    return bool(np.all(distances <= radius_nm))


def build_graph(
    df: pd.DataFrame,
    voyage: pd.DataFrame,
    config: Config,
    radius_nm: float,
    land_mask: LandMask | None,
) -> Graph:
    ref_lat = float(voyage["Lat"].mean())
    lat_min = float(voyage["Lat"].min() - config.corridor_margin_deg)
    lat_max = float(voyage["Lat"].max() + config.corridor_margin_deg)
    lon_min = float(voyage["Lon"].min() - config.corridor_margin_deg)
    lon_max = float(voyage["Lon"].max() + config.corridor_margin_deg)

    grid_lats = grid_centres(lat_min, lat_max, config.grid_step_deg)
    grid_lons = grid_centres(lon_min, lon_max, config.grid_step_deg)

    tree_df = df[
        (df["Lat"].between(lat_min - 1.0, lat_max + 1.0))
        & (df["Lon"].between(lon_min - 1.0, lon_max + 1.0))
    ]
    water_tree = build_observed_water_tree(tree_df, ref_lat)
    route_tree = build_observed_water_tree(voyage, ref_lat)

    lat_mesh, lon_mesh = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    node_xy = project_nm(lat_mesh.ravel(), lon_mesh.ravel(), ref_lat)
    dist_to_obs, _ = water_tree.query(node_xy, k=1)
    allowed = (dist_to_obs.reshape(lat_mesh.shape) <= radius_nm)
    if land_mask is not None:
        allowed_flat = allowed.ravel()
        for idx, (lat, lon) in enumerate(zip(lat_mesh.ravel(), lon_mesh.ravel())):
            land_here = land_mask.is_land(float(lon), float(lat))
            if land_here and not point_allowed(
                route_tree,
                float(lat),
                float(lon),
                ref_lat,
                config.land_override_radius_nm,
            ):
                allowed_flat[idx] = False

    node_to_ij: list[tuple[int, int]] = []
    ij_to_node: dict[tuple[int, int], int] = {}
    for i in range(len(grid_lats)):
        for j in range(len(grid_lons)):
            if allowed[i, j]:
                ij_to_node[(i, j)] = len(node_to_ij)
                node_to_ij.append((i, j))

    edge_rows: list[dict[str, float | int]] = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for (i, j), src in ij_to_node.items():
        for di, dj in directions:
            i2, j2 = i + di, j + dj
            if (i2, j2) not in ij_to_node:
                continue
            if di != 0 and dj != 0:
                if (i + di, j) not in ij_to_node or (i, j + dj) not in ij_to_node:
                    continue

            lat1, lon1 = float(grid_lats[i]), float(grid_lons[j])
            lat2, lon2 = float(grid_lats[i2]), float(grid_lons[j2])
            mid_lat, mid_lon = (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0
            if not point_allowed(water_tree, mid_lat, mid_lon, ref_lat, radius_nm):
                continue
            land_override_used = False
            if land_mask is not None and land_mask.segment_crosses_land(lon1, lat1, lon2, lat2):
                if not edge_near_track(
                    route_tree,
                    lat1,
                    lon1,
                    lat2,
                    lon2,
                    ref_lat,
                    config.land_override_radius_nm,
                ):
                    continue
                land_override_used = True

            edge_rows.append(
                {
                    "source": src,
                    "target": ij_to_node[(i2, j2)],
                    "lat1": lat1,
                    "lon1": lon1,
                    "lat2": lat2,
                    "lon2": lon2,
                    "mid_lat": mid_lat,
                    "mid_lon": mid_lon,
                    "distance_nm": haversine_nm(lat1, lon1, lat2, lon2),
                    "course_deg": course_deg(lat1, lon1, lat2, lon2),
                    "land_override": land_override_used,
                }
            )

    edges = pd.DataFrame(edge_rows)
    adjacency: list[list[int]] = [[] for _ in range(len(node_to_ij))]
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_idx, row in edges.iterrows():
        src, dst = int(row["source"]), int(row["target"])
        adjacency[src].append(edge_idx)
        edge_lookup[(src, dst)] = edge_idx

    start = voyage.iloc[0]
    end = voyage.iloc[-1]
    source = nearest_graph_node(float(start["Lat"]), float(start["Lon"]), node_to_ij, grid_lats, grid_lons, ref_lat)
    target = nearest_graph_node(float(end["Lat"]), float(end["Lon"]), node_to_ij, grid_lats, grid_lons, ref_lat)

    return Graph(
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        allowed=allowed,
        node_to_ij=node_to_ij,
        ij_to_node=ij_to_node,
        edges=edges,
        adjacency=adjacency,
        edge_lookup=edge_lookup,
        source=source,
        target=target,
        observed_radius_nm=radius_nm,
        land_mask_enabled=land_mask is not None,
    )


def attach_leg_nodes(graph: Graph, legs: list[Leg]) -> list[Leg]:
    ref_lat = float(np.mean(graph.grid_lats))
    attached = []
    for leg in legs:
        source = nearest_graph_node(
            leg.start_lat,
            leg.start_lon,
            graph.node_to_ij,
            graph.grid_lats,
            graph.grid_lons,
            ref_lat,
        )
        target = nearest_graph_node(
            leg.end_lat,
            leg.end_lon,
            graph.node_to_ij,
            graph.grid_lats,
            graph.grid_lons,
            ref_lat,
        )
        attached.append(
            Leg(
                leg_id=leg.leg_id,
                start_idx=leg.start_idx,
                end_idx=leg.end_idx,
                start_time=leg.start_time,
                end_time=leg.end_time,
                start_lat=leg.start_lat,
                start_lon=leg.start_lon,
                end_lat=leg.end_lat,
                end_lon=leg.end_lon,
                observed_distance_nm=leg.observed_distance_nm,
                straight_distance_nm=leg.straight_distance_nm,
                n_points=leg.n_points,
                source=source,
                target=target,
            )
        )
    return attached


def nearest_graph_node(
    lat: float,
    lon: float,
    node_to_ij: list[tuple[int, int]],
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    ref_lat: float,
) -> int:
    if not node_to_ij:
        raise ValueError("The route graph has no navigable nodes.")
    coords = np.array([(grid_lats[i], grid_lons[j]) for i, j in node_to_ij])
    tree = cKDTree(project_nm(coords[:, 0], coords[:, 1], ref_lat))
    _, idx = tree.query(project_nm(np.array([lat]), np.array([lon]), ref_lat), k=1)
    return int(idx[0])


def dijkstra(
    graph: Graph,
    edge_cost: np.ndarray,
    source: int | None = None,
    target: int | None = None,
) -> tuple[list[int], float]:
    source = graph.source if source is None else source
    target = graph.target if target is None else target

    dist = {source: 0.0}
    prev: dict[int, tuple[int, int] | None] = {source: None}
    heap = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)
        if node == target:
            break
        if cost > dist.get(node, float("inf")):
            continue
        for edge_idx in graph.adjacency[node]:
            dst = int(graph.edges.iloc[edge_idx]["target"])
            new_cost = cost + float(edge_cost[edge_idx])
            if new_cost < dist.get(dst, float("inf")):
                dist[dst] = new_cost
                prev[dst] = (node, edge_idx)
                heapq.heappush(heap, (new_cost, dst))

    if target not in prev:
        return [], float("inf")

    path = [target]
    cur = target
    while cur != source:
        parent = prev[cur]
        if parent is None:
            break
        cur = parent[0]
        path.append(cur)
    path.reverse()
    return path, dist[target]


def path_edge_ids(graph: Graph, path: list[int]) -> list[int]:
    return [graph.edge_lookup[(path[i], path[i + 1])] for i in range(len(path) - 1)]


def route_edge_ids(graph: Graph, leg_paths: list[list[int]]) -> list[int]:
    edge_ids: list[int] = []
    for path in leg_paths:
        edge_ids.extend(path_edge_ids(graph, path))
    return edge_ids


def route_node_count(leg_paths: list[list[int]]) -> int:
    return int(sum(len(path) for path in leg_paths))


def path_coords(graph: Graph, path: list[int]) -> list[tuple[float, float]]:
    coords = []
    for node in path:
        i, j = graph.node_to_ij[node]
        coords.append((float(graph.grid_lats[i]), float(graph.grid_lons[j])))
    return coords


class WeatherSampler:
    def __init__(self, df: pd.DataFrame, voyage_month: int, cell_deg: float, rng: np.random.Generator):
        self.cell_deg = cell_deg
        self.rng = rng
        months = month_window(voyage_month)
        seasonal = df[df["month"].isin(months)].copy()
        if len(seasonal) < 1000:
            seasonal = df.copy()

        seasonal["lat_cell"] = np.floor(seasonal["Lat"] / cell_deg).astype(int)
        seasonal["lon_cell"] = np.floor(seasonal["Lon"] / cell_deg).astype(int)
        self.groups: dict[tuple[int, int], np.ndarray] = {}
        variables = ["True_Wind_Speed_ms", "wind_dir_deg", "H_s", "uo", "vo"]
        for key, group in seasonal.groupby(["lat_cell", "lon_cell"], observed=True):
            self.groups[(int(key[0]), int(key[1]))] = group[variables].to_numpy(dtype=float)
        self.global_values = seasonal[variables].to_numpy(dtype=float)

        centres = []
        keys = []
        for key in self.groups:
            centres.append(((key[0] + 0.5) * cell_deg, (key[1] + 0.5) * cell_deg))
            keys.append(key)
        self.keys = keys
        self.tree = cKDTree(np.array(centres)) if centres else None

    def key_for(self, lat: float, lon: float) -> tuple[int, int]:
        return int(math.floor(lat / self.cell_deg)), int(math.floor(lon / self.cell_deg))

    def values_for(self, lat: float, lon: float) -> np.ndarray:
        key = self.key_for(lat, lon)
        values = self.groups.get(key)
        if values is not None and len(values) >= 10:
            return values
        if self.tree is not None:
            _, idx = self.tree.query(np.array([[lat, lon]]), k=1)
            nearest = self.groups[self.keys[int(idx[0])]]
            if len(nearest) >= 10:
                return nearest
        return self.global_values

    def sample(self, lat: float, lon: float, size: int) -> np.ndarray:
        values = self.values_for(lat, lon)
        sample_idx = self.rng.integers(0, len(values), size=size)
        return values[sample_idx]


def compute_edge_metrics(
    graph: Graph,
    sampler: WeatherSampler,
    config: Config,
    base_speed_kn: float,
) -> ScenarioMetrics:
    n_samples = config.samples
    n_edges = len(graph.edges)
    fuel_base = np.empty((n_samples, n_edges), dtype=np.float32)
    fuel_rotor = np.empty((n_samples, n_edges), dtype=np.float32)
    hours_no = np.empty((n_samples, n_edges), dtype=np.float32)
    hours_assist = np.empty((n_samples, n_edges), dtype=np.float32)
    rotor_energy = np.empty((n_samples, n_edges), dtype=np.float32)
    distance = graph.edges["distance_nm"].to_numpy(dtype=np.float32)

    for edge_idx, edge in graph.edges.iterrows():
        weather = sampler.sample(float(edge["mid_lat"]), float(edge["mid_lon"]), n_samples)
        wind_ms = weather[:, 0]
        wind_dir = weather[:, 1]
        hs = weather[:, 2]
        uo = weather[:, 3]
        vo = weather[:, 4]

        crs = float(edge["course_deg"])
        crs_rad = math.radians(crs)
        current_along_ms = uo * math.sin(crs_rad) + vo * math.cos(crs_rad)
        current_along_kn = current_along_ms / KN_TO_MS
        wave_loss = config.wave_loss_kn_per_m * np.maximum(0.0, hs - config.wave_calm_m)
        speed_no = np.clip(
            base_speed_kn + current_along_kn - wave_loss,
            config.min_speed_kn,
            config.max_speed_kn,
        )

        alpha_abs = np.abs(normalize_angle(wind_dir - crs))
        raw_rotor_kw = rotor_power_kw(wind_ms, alpha_abs)
        active = (raw_rotor_kw > 0.0) & (hs <= 6.0) & (wind_ms <= 42.0)
        p_rotor = np.where(active, raw_rotor_kw, 0.0)
        speed_assist = np.clip(
            speed_no + p_rotor / KW_PER_KN,
            config.min_speed_kn,
            config.max_speed_kn,
        )

        edge_dist = float(edge["distance_nm"])
        edge_hours_no = edge_dist / speed_no
        edge_hours_assist = edge_dist / speed_assist
        p_brake = brake_power_kw(speed_no)
        p_net = np.maximum(p_brake - p_rotor, 0.0)

        fuel_base[:, edge_idx] = fuel_rate_t_h(p_brake) * edge_hours_no
        fuel_rotor[:, edge_idx] = fuel_rate_t_h(p_net) * edge_hours_no
        hours_no[:, edge_idx] = edge_hours_no
        hours_assist[:, edge_idx] = edge_hours_assist
        rotor_energy[:, edge_idx] = p_rotor * edge_hours_no

    return ScenarioMetrics(
        fuel_base_t=fuel_base,
        fuel_rotor_t=fuel_rotor,
        hours_no_rotor=hours_no,
        hours_speed_assist=hours_assist,
        rotor_energy_kwh=rotor_energy,
        distance_nm=distance,
    )


def evaluate_observed_route(
    name: str,
    voyage: pd.DataFrame,
    legs: list[Leg],
    sampler: WeatherSampler,
    config: Config,
    base_speed_kn: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Evaluate the actually sailed AIS route under the Monte Carlo model."""

    n_samples = config.samples
    fuel_base = np.zeros(n_samples, dtype=np.float64)
    fuel_rotor = np.zeros(n_samples, dtype=np.float64)
    hours_no = np.zeros(n_samples, dtype=np.float64)
    hours_assist = np.zeros(n_samples, dtype=np.float64)
    rotor_energy = np.zeros(n_samples, dtype=np.float64)
    total_distance = 0.0
    path_nodes = 0

    for leg in legs:
        segment = voyage.iloc[leg.start_idx : leg.end_idx + 1].reset_index(drop=True)
        path_nodes += len(segment)
        for idx in range(1, len(segment)):
            prev = segment.iloc[idx - 1]
            cur = segment.iloc[idx]
            edge_dist = float(cur.get("Dist_since_last_nm", np.nan))
            if not np.isfinite(edge_dist) or edge_dist <= 0.0:
                edge_dist = haversine_nm(float(prev["Lat"]), float(prev["Lon"]), float(cur["Lat"]), float(cur["Lon"]))
            if edge_dist <= 0.0:
                continue

            crs = course_deg(float(prev["Lat"]), float(prev["Lon"]), float(cur["Lat"]), float(cur["Lon"]))
            mid_lat = float((prev["Lat"] + cur["Lat"]) / 2.0)
            mid_lon = float((prev["Lon"] + cur["Lon"]) / 2.0)
            weather = sampler.sample(mid_lat, mid_lon, n_samples)

            wind_ms = weather[:, 0]
            wind_dir = weather[:, 1]
            hs = weather[:, 2]
            uo = weather[:, 3]
            vo = weather[:, 4]

            crs_rad = math.radians(crs)
            current_along_ms = uo * math.sin(crs_rad) + vo * math.cos(crs_rad)
            current_along_kn = current_along_ms / KN_TO_MS
            wave_loss = config.wave_loss_kn_per_m * np.maximum(0.0, hs - config.wave_calm_m)
            speed_no = np.clip(
                base_speed_kn + current_along_kn - wave_loss,
                config.min_speed_kn,
                config.max_speed_kn,
            )

            alpha_abs = np.abs(normalize_angle(wind_dir - crs))
            raw_rotor_kw = rotor_power_kw(wind_ms, alpha_abs)
            active = (raw_rotor_kw > 0.0) & (hs <= 6.0) & (wind_ms <= 42.0)
            p_rotor = np.where(active, raw_rotor_kw, 0.0)
            speed_assist = np.clip(
                speed_no + p_rotor / KW_PER_KN,
                config.min_speed_kn,
                config.max_speed_kn,
            )

            edge_hours_no = edge_dist / speed_no
            edge_hours_assist = edge_dist / speed_assist
            p_brake = brake_power_kw(speed_no)
            p_net = np.maximum(p_brake - p_rotor, 0.0)

            fuel_base += fuel_rate_t_h(p_brake) * edge_hours_no
            fuel_rotor += fuel_rate_t_h(p_net) * edge_hours_no
            hours_no += edge_hours_no
            hours_assist += edge_hours_assist
            rotor_energy += p_rotor * edge_hours_no
            total_distance += edge_dist

    fuel_saved = fuel_base - fuel_rotor
    co2_saved = fuel_saved * EF_HFO
    mean_rotor_kw = rotor_energy / np.maximum(hours_no, 1e-9)

    per_scenario = pd.DataFrame(
        {
            "route": name,
            "scenario": np.arange(n_samples),
            "distance_nm": total_distance,
            "fuel_base_t": fuel_base,
            "fuel_rotor_t": fuel_rotor,
            "fuel_saved_t": fuel_saved,
            "co2_saved_t": co2_saved,
            "eta_no_rotor_h": hours_no,
            "eta_speed_assist_h": hours_assist,
            "mean_rotor_kw": mean_rotor_kw,
            "path_nodes": path_nodes,
            "legs": len(legs),
        }
    )

    summary: dict[str, float | str] = {
        "route": name,
        "distance_nm": total_distance,
        "path_nodes": path_nodes,
        "legs": len(legs),
    }
    for col in [
        "fuel_base_t",
        "fuel_rotor_t",
        "fuel_saved_t",
        "co2_saved_t",
        "eta_no_rotor_h",
        "eta_speed_assist_h",
        "mean_rotor_kw",
    ]:
        summary.update(summarize_array(per_scenario[col].to_numpy(), col))
    return per_scenario, summary


def summarize_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
    }


def evaluate_fixed_route(
    name: str,
    graph: Graph,
    leg_paths: list[list[int]],
    metrics: ScenarioMetrics,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    edge_ids = route_edge_ids(graph, leg_paths)
    fuel_base = metrics.fuel_base_t[:, edge_ids].sum(axis=1)
    fuel_rotor = metrics.fuel_rotor_t[:, edge_ids].sum(axis=1)
    hours_no = metrics.hours_no_rotor[:, edge_ids].sum(axis=1)
    hours_assist = metrics.hours_speed_assist[:, edge_ids].sum(axis=1)
    rotor_energy = metrics.rotor_energy_kwh[:, edge_ids].sum(axis=1)
    distance = float(metrics.distance_nm[edge_ids].sum())
    fuel_saved = fuel_base - fuel_rotor
    co2_saved = fuel_saved * EF_HFO
    mean_rotor_kw = rotor_energy / np.maximum(hours_no, 1e-9)

    per_scenario = pd.DataFrame(
        {
            "route": name,
            "scenario": np.arange(len(fuel_base)),
            "distance_nm": distance,
            "fuel_base_t": fuel_base,
            "fuel_rotor_t": fuel_rotor,
            "fuel_saved_t": fuel_saved,
            "co2_saved_t": co2_saved,
            "eta_no_rotor_h": hours_no,
            "eta_speed_assist_h": hours_assist,
            "mean_rotor_kw": mean_rotor_kw,
            "path_nodes": route_node_count(leg_paths),
            "legs": len(leg_paths),
        }
    )

    summary: dict[str, float | str] = {
        "route": name,
        "distance_nm": distance,
        "path_nodes": route_node_count(leg_paths),
        "legs": len(leg_paths),
    }
    for col in [
        "fuel_base_t",
        "fuel_rotor_t",
        "fuel_saved_t",
        "co2_saved_t",
        "eta_no_rotor_h",
        "eta_speed_assist_h",
        "mean_rotor_kw",
    ]:
        summary.update(summarize_array(per_scenario[col].to_numpy(), col))
    return per_scenario, summary


def evaluate_scenario_optimal(
    graph: Graph,
    paths: list[list[list[int]]],
    metrics: ScenarioMetrics,
) -> tuple[pd.DataFrame, dict[str, float | str], dict[int, int]]:
    rows = []
    edge_counts: dict[int, int] = {}
    for scenario, leg_paths in enumerate(paths):
        edge_ids = route_edge_ids(graph, leg_paths)
        for edge_id in edge_ids:
            edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1

        fuel_base = float(metrics.fuel_base_t[scenario, edge_ids].sum())
        fuel_rotor = float(metrics.fuel_rotor_t[scenario, edge_ids].sum())
        hours_no = float(metrics.hours_no_rotor[scenario, edge_ids].sum())
        hours_assist = float(metrics.hours_speed_assist[scenario, edge_ids].sum())
        rotor_energy = float(metrics.rotor_energy_kwh[scenario, edge_ids].sum())
        distance = float(metrics.distance_nm[edge_ids].sum())
        fuel_saved = fuel_base - fuel_rotor
        rows.append(
            {
                "route": "Scenario-optimal upper bound",
                "scenario": scenario,
                "distance_nm": distance,
                "fuel_base_t": fuel_base,
                "fuel_rotor_t": fuel_rotor,
                "fuel_saved_t": fuel_saved,
                "co2_saved_t": fuel_saved * EF_HFO,
                "eta_no_rotor_h": hours_no,
                "eta_speed_assist_h": hours_assist,
                "mean_rotor_kw": rotor_energy / max(hours_no, 1e-9),
                "path_nodes": route_node_count(leg_paths),
                "legs": len(leg_paths),
            }
        )

    per_scenario = pd.DataFrame(rows)
    summary: dict[str, float | str] = {"route": "Scenario-optimal upper bound"}
    summary["distance_nm"] = float(per_scenario["distance_nm"].mean())
    summary["path_nodes"] = float(per_scenario["path_nodes"].mean())
    summary["legs"] = float(per_scenario["legs"].mean())
    for col in [
        "fuel_base_t",
        "fuel_rotor_t",
        "fuel_saved_t",
        "co2_saved_t",
        "eta_no_rotor_h",
        "eta_speed_assist_h",
        "mean_rotor_kw",
    ]:
        summary.update(summarize_array(per_scenario[col].to_numpy(), col))
    return per_scenario, summary, edge_counts


def build_routes(
    graph: Graph,
    legs: list[Leg],
    metrics: ScenarioMetrics,
    config: Config,
) -> tuple[dict[str, list[list[int]]], list[list[list[int]]]]:
    routes: dict[str, list[list[int]]] = {
        "Shortest feasible": [],
        "Expected-cost MC": [],
        "Risk-aware MC": [],
    }
    scenario_paths: list[list[list[int]]] = [[] for _ in range(config.samples)]

    distance_cost = graph.edges["distance_nm"].to_numpy(dtype=float)
    expected_cost = metrics.fuel_rotor_t.mean(axis=0).astype(float)
    risk_cost = expected_cost + config.risk_lambda * metrics.fuel_rotor_t.std(axis=0).astype(float)

    for leg in legs:
        if leg.source is None or leg.target is None:
            raise RuntimeError(f"Leg {leg.leg_id} has not been attached to graph nodes.")

        shortest_path, _ = dijkstra(graph, distance_cost, leg.source, leg.target)
        if not shortest_path:
            raise RuntimeError(f"No feasible sea-graph path was found for leg {leg.leg_id}.")

        expected_path, _ = dijkstra(graph, expected_cost, leg.source, leg.target)
        risk_path, _ = dijkstra(graph, risk_cost, leg.source, leg.target)

        routes["Shortest feasible"].append(shortest_path)
        routes["Expected-cost MC"].append(expected_path or shortest_path)
        routes["Risk-aware MC"].append(risk_path or shortest_path)

        for scenario in range(config.samples):
            path, _ = dijkstra(graph, metrics.fuel_rotor_t[scenario].astype(float), leg.source, leg.target)
            scenario_paths[scenario].append(path or shortest_path)

    return routes, scenario_paths


def make_path_table(graph: Graph, routes: dict[str, list[list[int]]]) -> pd.DataFrame:
    rows = []
    for route, leg_paths in routes.items():
        for leg_id, path in enumerate(leg_paths, start=1):
            for order, (lat, lon) in enumerate(path_coords(graph, path)):
                rows.append({"route": route, "leg_id": leg_id, "order": order, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def make_scenario_path_table(graph: Graph, scenario_paths: list[list[list[int]]]) -> pd.DataFrame:
    rows = []
    for scenario, leg_paths in enumerate(scenario_paths):
        for leg_id, path in enumerate(leg_paths, start=1):
            for order, (lat, lon) in enumerate(path_coords(graph, path)):
                rows.append(
                    {
                        "scenario": scenario,
                        "leg_id": leg_id,
                        "order": order,
                        "lat": lat,
                        "lon": lon,
                    }
                )
    return pd.DataFrame(rows)


def edge_frequency_table(graph: Graph, edge_counts: dict[int, int], samples: int) -> pd.DataFrame:
    rows = []
    for edge_idx, count in edge_counts.items():
        edge = graph.edges.iloc[edge_idx]
        rows.append(
            {
                "edge_idx": edge_idx,
                "count": count,
                "frequency": count / samples,
                "lat1": edge["lat1"],
                "lon1": edge["lon1"],
                "lat2": edge["lat2"],
                "lon2": edge["lon2"],
                "land_override": bool(edge.get("land_override", False)),
            }
        )
    return pd.DataFrame(rows).sort_values("frequency", ascending=False)


def plot_outputs(
    df: pd.DataFrame,
    voyage: pd.DataFrame,
    graph: Graph,
    routes: dict[str, list[list[int]]],
    scenario_paths: list[list[list[int]]],
    scenario_plot_count: int,
    all_metrics: pd.DataFrame,
    edge_freq: pd.DataFrame,
    stops: pd.DataFrame,
    out_path: Path,
) -> None:
    route_order = [
        "Original observed route",
        "Shortest feasible",
        "Expected-cost MC",
        "Risk-aware MC",
        "Scenario-optimal upper bound",
    ]
    colors = {
        "Original observed route": "#111111",
        "Shortest feasible": "#356AA0",
        "Expected-cost MC": "#2D8A49",
        "Risk-aware MC": "#C46A1B",
        "Scenario-optimal upper bound": "#7A4DB3",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    sample_df = df.sample(min(len(df), 6000), random_state=7)
    ax.scatter(sample_df["Lon"], sample_df["Lat"], s=2, c="lightgray", alpha=0.25, label="Historical AIS")
    ax.plot(voyage["Lon"], voyage["Lat"], color="black", linewidth=1.2, alpha=0.6, label="Reference voyage")

    if not edge_freq.empty:
        top = edge_freq[edge_freq["frequency"] >= max(0.04, edge_freq["frequency"].quantile(0.80))]
        segments = [[(r["lon1"], r["lat1"]), (r["lon2"], r["lat2"])] for _, r in top.iterrows()]
        widths = 0.5 + 5.0 * top["frequency"].to_numpy()
        lc = LineCollection(segments, colors=colors["Scenario-optimal upper bound"], linewidths=widths, alpha=0.18)
        ax.add_collection(lc)

    if scenario_paths and scenario_plot_count > 0:
        scenario_indices = np.linspace(
            0,
            len(scenario_paths) - 1,
            min(scenario_plot_count, len(scenario_paths)),
            dtype=int,
        )
        scenario_labelled = False
        for scenario_idx in scenario_indices:
            for path in scenario_paths[int(scenario_idx)]:
                coords = path_coords(graph, path)
                lats = [lat for lat, _ in coords]
                lons = [lon for _, lon in coords]
                ax.plot(
                    lons,
                    lats,
                    color=colors["Scenario-optimal upper bound"],
                    alpha=0.16,
                    linewidth=1.0,
                    label="Sample MC scenario route" if not scenario_labelled else None,
                    zorder=2,
                )
                scenario_labelled = True

    for route, leg_paths in routes.items():
        labelled = False
        for path in leg_paths:
            coords = path_coords(graph, path)
            lats = [lat for lat, _ in coords]
            lons = [lon for _, lon in coords]
            ax.plot(
                lons,
                lats,
                linewidth=2.3,
                color=colors[route],
                label=route if not labelled else None,
            )
            ax.scatter([lons[0], lons[-1]], [lats[0], lats[-1]], s=24, color=colors[route])
            labelled = True

    if not stops.empty:
        ax.scatter(
            stops["stop_lon"],
            stops["stop_lat"],
            marker="s",
            s=46,
            color="#9A1F40",
            edgecolor="white",
            linewidth=0.6,
            label="Detected stop gap",
            zorder=6,
        )

    mask_label = "land-aware sea graph" if graph.land_mask_enabled else "observed-water graph"
    ax.set_title(f"Feasible routes on {mask_label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    plot_data = [all_metrics.loc[all_metrics["route"] == r, "fuel_rotor_t"].to_numpy() for r in route_order]
    ax.boxplot(plot_data, tick_labels=route_order, showfliers=False)
    ax.set_ylabel("Fuel with rotor [t]")
    ax.set_title("Monte Carlo fuel-use distribution")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    for route in route_order:
        vals = np.sort(all_metrics.loc[all_metrics["route"] == route, "co2_saved_t"].to_numpy())
        y = np.linspace(0, 1, len(vals), endpoint=True)
        ax.plot(vals, y, label=route, color=colors.get(route, "gray"), linewidth=1.8)
    ax.set_xlabel("CO2 saved vs no-rotor operation [t]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CO2 saving confidence distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for route in route_order:
        vals = np.sort(all_metrics.loc[all_metrics["route"] == route, "eta_speed_assist_h"].to_numpy())
        y = np.linspace(0, 1, len(vals), endpoint=True)
        ax.plot(vals, y, label=route, color=colors.get(route, "gray"), linewidth=1.8)
    ax.set_xlabel("ETA with speed-assist interpretation [h]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("ETA confidence distribution")
    ax.legend(fontsize=8)

    fig.suptitle("Monte Carlo route optimisation under weather uncertainty", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    summary: pd.DataFrame,
    all_metrics: pd.DataFrame,
    path_table: pd.DataFrame,
    scenario_path_table: pd.DataFrame,
    edge_freq: pd.DataFrame,
    graph: Graph,
    legs: list[Leg],
    stops: pd.DataFrame,
    voyage_id: int,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "monte_carlo_route_summary.csv", index=False)
    all_metrics.to_csv(RESULTS_DIR / "monte_carlo_route_metrics.csv", index=False)
    path_table.to_csv(RESULTS_DIR / "monte_carlo_route_paths.csv", index=False)
    scenario_path_table.to_csv(RESULTS_DIR / "monte_carlo_scenario_route_paths.csv", index=False)
    edge_freq.to_csv(RESULTS_DIR / "monte_carlo_edge_frequency.csv", index=False)
    legs_df = pd.DataFrame(
        [
            {
                "leg_id": leg.leg_id,
                "start_time": leg.start_time,
                "end_time": leg.end_time,
                "start_lat": leg.start_lat,
                "start_lon": leg.start_lon,
                "end_lat": leg.end_lat,
                "end_lon": leg.end_lon,
                "observed_distance_nm": leg.observed_distance_nm,
                "straight_distance_nm": leg.straight_distance_nm,
                "n_points": leg.n_points,
            }
            for leg in legs
        ]
    )
    legs_df.to_csv(RESULTS_DIR / "monte_carlo_itinerary_legs.csv", index=False)
    stops.to_csv(RESULTS_DIR / "monte_carlo_detected_stops.csv", index=False)

    txt = RESULTS_DIR / "monte_carlo_route_summary.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write(f"Monte Carlo route optimisation summary for Voyage_ID {voyage_id}\n")
        f.write("=" * 72 + "\n")
        f.write(f"Land mask enabled: {graph.land_mask_enabled}\n")
        f.write(f"Graph nodes/edges: {len(graph.node_to_ij)} / {len(graph.edges)}\n")
        if graph.land_mask_enabled and "land_override" in graph.edges:
            f.write(f"AIS-corridor override edges: {int(graph.edges['land_override'].sum())}\n")
        f.write(f"Detected sailing legs: {len(legs)}\n")
        f.write(f"Detected stop gaps: {len(stops)}\n")
        f.write("\nItinerary legs forced through stop/departure points:\n")
        leg_cols = [
            "leg_id",
            "start_time",
            "end_time",
            "observed_distance_nm",
            "straight_distance_nm",
            "n_points",
        ]
        leg_print = legs_df[leg_cols].copy()
        for col in ["observed_distance_nm", "straight_distance_nm"]:
            leg_print[col] = leg_print[col].round(3)
        f.write(leg_print.to_string(index=False))
        f.write("\n")
        if not stops.empty:
            f.write("\nDetected stop gaps:\n")
            stop_cols = [
                "stop_id",
                "arrival_time",
                "departure_time",
                "gap_hours",
                "gap_distance_nm",
                "stop_lat",
                "stop_lon",
            ]
            stop_print = stops[stop_cols].copy()
            for col in ["gap_hours", "gap_distance_nm", "stop_lat", "stop_lon"]:
                stop_print[col] = stop_print[col].round(3)
            f.write(stop_print.to_string(index=False))
            f.write("\n")
        f.write("\nRoute confidence summary:\n")
        keep = [
            "route",
            "legs",
            "distance_nm",
            "fuel_rotor_t_mean",
            "fuel_rotor_t_p05",
            "fuel_rotor_t_p95",
            "fuel_saved_t_mean",
            "co2_saved_t_mean",
            "eta_speed_assist_h_mean",
            "prob_beats_shortest_fuel",
        ]
        f.write(summary[[c for c in keep if c in summary.columns]].round(3).to_string(index=False))
        f.write("\n")


def main() -> None:
    config = parse_args()
    rng = np.random.default_rng(config.seed)

    print("Loading engineered dataset...")
    df = load_data(DATA_PATH)
    voyage_id = select_reference_voyage(df, config.voyage_id)
    voyage = df[df["Voyage_ID"] == voyage_id].copy().reset_index(drop=True)
    legs_raw, stops = detect_itinerary_legs(voyage, config.stop_gap_hours)
    base_speed = float(config.base_speed_kn or df["Speed_kn"].median())

    print(f"Selected Voyage_ID: {voyage_id}")
    print(f"Reference voyage points: {len(voyage):,}")
    print(
        "Start/end: "
        f"({voyage.iloc[0]['Lat']:.3f}, {voyage.iloc[0]['Lon']:.3f}) -> "
        f"({voyage.iloc[-1]['Lat']:.3f}, {voyage.iloc[-1]['Lon']:.3f})"
    )
    print(
        f"Detected {len(legs_raw)} sailing legs and {len(stops)} stop gaps "
        f"(gap threshold > {config.stop_gap_hours:.1f} h)"
    )
    for leg in legs_raw:
        print(
            f"  Leg {leg.leg_id}: {leg.start_time} -> {leg.end_time}, "
            f"observed {leg.observed_distance_nm:.1f} nm, "
            f"straight {leg.straight_distance_nm:.1f} nm, {leg.n_points} points"
        )
    print(f"Base speed: {base_speed:.2f} kn")

    land_mask = None
    if not config.disable_land_mask:
        bounds = route_bounds(voyage, config.corridor_margin_deg)
        print(f"Loading Natural Earth land mask: {config.land_geojson_path}")
        land_mask = LandMask.from_geojson(config.land_geojson_path, bounds=bounds)
        print(
            f"Land mask: {len(land_mask.polygons)} polygons, "
            f"{len(land_mask.seg_lon1):,} coastline segments in route bounds, "
            f"AIS corridor override {config.land_override_radius_nm:.1f} nm"
        )
    else:
        print("Land mask disabled; using observed-water proximity only.")

    graph = None
    legs: list[Leg] = []
    radius = config.observed_water_radius_nm
    while radius <= config.max_water_radius_nm:
        candidate = build_graph(df, voyage, config, radius, land_mask)
        candidate_legs = attach_leg_nodes(candidate, legs_raw)
        distance_cost = candidate.edges["distance_nm"].to_numpy(dtype=float)
        leg_paths = [
            dijkstra(candidate, distance_cost, leg.source, leg.target)[0]
            for leg in candidate_legs
            if leg.source is not None and leg.target is not None
        ]
        if len(leg_paths) == len(candidate_legs) and all(leg_paths):
            graph = candidate
            legs = candidate_legs
            break
        radius += 15.0
    if graph is None:
        raise RuntimeError("Could not build a connected observed-water graph for all itinerary legs.")

    print(
        f"Graph: {len(graph.node_to_ij):,} navigable nodes, "
        f"{len(graph.edges):,} directed edges, observed-water radius {graph.observed_radius_nm:.1f} nm, "
        f"land mask {'on' if graph.land_mask_enabled else 'off'}"
    )
    if graph.land_mask_enabled and "land_override" in graph.edges:
        print(f"AIS-corridor override edges: {int(graph.edges['land_override'].sum())}")

    print("Sampling weather and computing edge metrics...")
    voyage_month = int(voyage.iloc[0]["month"])
    sampler = WeatherSampler(df, voyage_month, config.weather_cell_deg, rng)
    metrics = compute_edge_metrics(graph, sampler, config, base_speed)

    print("Optimising routes across Monte Carlo scenarios...")
    routes, scenario_paths = build_routes(graph, legs, metrics, config)

    observed_metrics, observed_summary = evaluate_observed_route(
        "Original observed route",
        voyage,
        legs,
        sampler,
        config,
        base_speed,
    )
    fixed_metrics = [observed_metrics]
    summaries = [observed_summary]
    for route, path in routes.items():
        route_metrics, route_summary = evaluate_fixed_route(route, graph, path, metrics)
        fixed_metrics.append(route_metrics)
        summaries.append(route_summary)

    scenario_metrics, scenario_summary, edge_counts = evaluate_scenario_optimal(graph, scenario_paths, metrics)
    fixed_metrics.append(scenario_metrics)
    summaries.append(scenario_summary)

    all_metrics = pd.concat(fixed_metrics, ignore_index=True)
    summary = pd.DataFrame(summaries)

    shortest = all_metrics[all_metrics["route"] == "Shortest feasible"].set_index("scenario")["fuel_rotor_t"]
    probs = []
    for route in summary["route"]:
        route_vals = all_metrics[all_metrics["route"] == route].set_index("scenario")["fuel_rotor_t"]
        aligned = route_vals.reindex(shortest.index)
        probs.append(float((aligned < shortest).mean()))
    summary["prob_beats_shortest_fuel"] = probs

    edge_freq = edge_frequency_table(graph, edge_counts, config.samples)
    path_table = make_path_table(graph, routes)
    scenario_path_table = make_scenario_path_table(graph, scenario_paths)

    print("Writing outputs...")
    write_outputs(summary, all_metrics, path_table, scenario_path_table, edge_freq, graph, legs, stops, voyage_id)
    plot_outputs(
        df=df,
        voyage=voyage,
        graph=graph,
        routes=routes,
        scenario_paths=scenario_paths,
        scenario_plot_count=config.scenario_plot_count,
        all_metrics=all_metrics,
        edge_freq=edge_freq,
        stops=stops,
        out_path=RESULTS_DIR / "monte_carlo_routing.png",
    )

    display_cols = [
        "route",
        "legs",
        "distance_nm",
        "fuel_rotor_t_mean",
        "fuel_saved_t_mean",
        "co2_saved_t_mean",
        "eta_speed_assist_h_mean",
        "prob_beats_shortest_fuel",
    ]
    print()
    print("Monte Carlo route summary")
    print(summary[display_cols].round(3).to_string(index=False))
    print()
    print(f"Saved: {RESULTS_DIR / 'monte_carlo_routing.png'}")
    print(f"Saved: {RESULTS_DIR / 'monte_carlo_route_summary.csv'}")


if __name__ == "__main__":
    main()
