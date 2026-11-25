"""Tool to display and explore the NAYRA Object Location Memory Database.

Renders an interactive CLI table, summary statistics of remembered items,
and automatically exports an interactive HTML dashboard for video demonstration.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sqlite3
import sys
import webbrowser

root_dir = Path(__file__).resolve().parents[1]
db_path = root_dir / "data" / "memory" / "object_memory.db"


def get_latest_objects_summary(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            canonical_name,
            COUNT(*) as total_sightings,
            MAX(timestamp) as last_seen_ts,
            MAX(iso_time) as last_iso,
            horizontal_zone,
            spatial_relation,
            anchor_object,
            ROUND(confidence * 100, 1) as conf_pct
        FROM object_observations
        GROUP BY canonical_name
        ORDER BY last_seen_ts DESC;
    """)
    return cursor.fetchall()


def get_recent_history(conn: sqlite3.Connection, limit: int = 20):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            id,
            canonical_name,
            iso_time,
            horizontal_zone,
            spatial_relation,
            anchor_object,
            ROUND(confidence * 100, 1) as conf_pct
        FROM object_observations
        ORDER BY timestamp DESC
        LIMIT ?;
    """, (limit,))
    return cursor.fetchall()


def generate_html_dashboard(latest_items, recent_history, total_count: int, out_html_path: Path):
    """Generate a sleek, dark-mode visual web dashboard for presentation."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    
    rows_unique = ""
    for r in latest_items:
        name, sightings, last_ts, last_iso, zone, relation, anchor, conf = r
        rel_str = f"{relation} {anchor}" if relation and anchor else (zone or "In view")
        badge_color = "#10b981" if zone == "CENTER" else ("#3b82f6" if zone == "LEFT" else "#f59e0b")
        rows_unique += f"""
        <tr>
            <td style="font-weight: 600; text-transform: uppercase; color: #38bdf8;">{name}</td>
            <td><span class="badge" style="background: {badge_color};">{zone or 'CENTER'}</span></td>
            <td>{rel_str}</td>
            <td style="color: #94a3b8;">{last_iso.replace('T', ' ')[:19]}</td>
            <td><span class="count-pill">{sightings} times</span></td>
            <td style="color: #4ade80;">{conf}%</td>
        </tr>
        """

    rows_recent = ""
    for r in recent_history:
        oid, name, iso, zone, relation, anchor, conf = r
        rel_str = f"{relation} {anchor}" if relation and anchor else (zone or "In view")
        rows_recent += f"""
        <tr>
            <td style="color: #64748b;">#{oid}</td>
            <td style="font-weight: 600; text-transform: uppercase;">{name}</td>
            <td>{zone or 'CENTER'}</td>
            <td>{rel_str}</td>
            <td style="color: #94a3b8;">{iso.replace('T', ' ')[:19]}</td>
            <td style="color: #4ade80;">{conf}%</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NAYRA - Object Location Memory Database</title>
    <style>
        body {{
            background-color: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #1e293b;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .title {{
            font-size: 28px;
            font-weight: bold;
            color: #38bdf8;
            letter-spacing: 0.5px;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
            margin-top: 5px;
        }}
        .stat-card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 25px;
            min-width: 200px;
            border: 1px solid #334155;
        }}
        .stat-val {{
            font-size: 32px;
            font-weight: bold;
            color: #38bdf8;
        }}
        .stat-lbl {{
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 40px;
            border: 1px solid #334155;
        }}
        th, td {{
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid #334155;
            font-size: 14px;
        }}
        th {{
            background: #0f172a;
            color: #94a3b8;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: #243248;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: bold;
            color: #ffffff;
            text-transform: uppercase;
        }}
        .count-pill {{
            background: #334155;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            color: #cbd5e1;
        }}
        h2 {{
            color: #e2e8f0;
            font-size: 18px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">🧠 NAYRA - Object Location Memory Database</div>
            <div class="subtitle">Real-Time Persistent Spatial Observation Archive | Module 10</div>
        </div>
        <div style="text-align: right; color: #94a3b8; font-size: 13px;">
            Exported: {now_str}<br>
            Database: <code>data/memory/object_memory.db</code>
        </div>
    </div>

    <div>
        <div class="stat-card">
            <div class="stat-val">{total_count}</div>
            <div class="stat-lbl">Total Observations Recorded</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="color: #4ade80;">{len(latest_items)}</div>
            <div class="stat-lbl">Unique Distinct Objects</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="color: #f59e0b;">SQLite WAL</div>
            <div class="stat-lbl">Storage Architecture</div>
        </div>
    </div>

    <h2>📍 Distinct Objects Remembered (Latest Known Locations)</h2>
    <table>
        <thead>
            <tr>
                <th>Object Name</th>
                <th>Horizontal Zone</th>
                <th>Spatial Relationship</th>
                <th>Last Seen Timestamp</th>
                <th>Sightings</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {rows_unique}
        </tbody>
    </table>

    <h2>⏱️ Recent Temporal Observations Log (Latest 20 Sightings)</h2>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Object</th>
                <th>Zone</th>
                <th>Spatial Context</th>
                <th>Observation Time</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {rows_recent}
        </tbody>
    </table>
</body>
</html>
"""
    out_html_path.write_text(html, encoding="utf-8")
    return out_html_path


def main():
    parser = argparse.ArgumentParser(description="View NAYRA Object Memory Database")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent records to display")
    parser.add_argument("--html", action="store_true", help="Open visual HTML dashboard in web browser")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch browser")
    args, _ = parser.parse_known_args()

    if not db_path.exists():
        print(f"[ERROR] Database file not found at: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM object_observations;")
    total_count = cursor.fetchone()[0]

    latest_items = get_latest_objects_summary(conn)
    recent_history = get_recent_history(conn, limit=args.limit)

    print("\n" + "=" * 90)
    print(f"  🧠 NAYRA OBJECT LOCATION MEMORY DATABASE (Total Sightings: {total_count})")
    print("=" * 90)
    print(f"{'OBJECT':<16} | {'ZONE':<8} | {'LAST SEEN (ISO)':<20} | {'SIGHTINGS':<10} | {'SPATIAL RELATION'}")
    print("-" * 90)

    for r in latest_items:
        name, sightings, last_ts, last_iso, zone, relation, anchor, conf = r
        rel_str = f"{relation} {anchor}" if relation and anchor else (zone or "In view")
        time_short = last_iso.replace("T", " ")[:19]
        print(f"{name.upper():<16} | {(zone or 'CENTER'):<8} | {time_short:<20} | {sightings:<10} | {rel_str}")

    print("=" * 90)

    # Export and open HTML dashboard
    export_dir = root_dir / "recordings"
    export_dir.mkdir(exist_ok=True)
    html_file = export_dir / "object_memory_dashboard.html"
    generate_html_dashboard(latest_items, recent_history, total_count, html_file)
    print(f"\n[DASHBOARD EXPORTED]: file:///{html_file.as_posix()}")

    if not args.no_browser or args.html:
        print("[BROWSER] Launching visual memory dashboard in browser...")
        webbrowser.open(html_file.as_uri())

    conn.close()


if __name__ == "__main__":
    main()
