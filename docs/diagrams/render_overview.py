"""Build the presentation overview as editable SVG and high-resolution PNG."""

from html import escape
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def render(dark=False):
    p = ({"bg": "#111024", "ink": "#f8f6ff", "muted": "#b8b3d1", "line": "#9282c9",
          "card": "#201d3c", "purple_bg": "#302452", "teal_bg": "#163e47", "border": "#50486e",
          "white": "#f7f4ff", "purple": "#bc9bff", "teal": "#6fe4d5", "panel": "#19172f"}
         if dark else
         {"bg": "#fbfaff", "ink": "#1e1243", "muted": "#665c7d", "line": "#8872b5",
          "card": "#ffffff", "purple_bg": "#f0eafd", "teal_bg": "#e1f7f3", "border": "#e3dcf0",
          "white": "#ffffff", "purple": "#7041db", "teal": "#138f84", "panel": "#f5f2fb"})
    purple, teal = p["purple"], p["teal"]
    s = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1360" viewBox="0 0 2400 1360" role="img" aria-labelledby="title desc">
<title id="title">BRAMS AI Assistant system architecture</title>
<desc id="desc">Gradio chat exchanges questions and answers with a Pydantic AI agent, which uses a language model and validated BRAMS tools. The tools query the external archive and invoke a local meteor detector. Local media and annotated images are displayed in the workspace. Grounding, run limits and recovery safeguard the turn.</desc>
<defs>
  <filter id="shadow" x="-15%" y="-20%" width="130%" height="150%"><feDropShadow dx="0" dy="8" stdDeviation="14" flood-color="#2b1850" flood-opacity=".08"/></filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{purple}"/></marker>
  <marker id="arrow-teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{teal}"/></marker>
</defs>
<rect width="2400" height="1360" fill="{p['bg']}"/>
<g font-family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" fill="{p['ink']}">
<rect x="80" y="52" width="210" height="38" rx="19" fill="{p['purple_bg']}"/>
<text x="105" y="77" font-size="16" letter-spacing="2.2" font-weight="750" fill="{purple}">SYSTEM OVERVIEW</text>
<text x="80" y="158" font-size="62" font-weight="780" letter-spacing="-2">BRAMS AI Assistant</text>
<text x="82" y="201" font-size="24" fill="{p['muted']}">Natural-language research, archive evidence and local meteor echo detection</text>
<rect x="1745" y="83" width="230" height="54" rx="27" fill="{p['purple_bg']}"/><text x="1780" y="117" font-size="19" font-weight="700" fill="{purple}">12 agent tools</text>
<rect x="1995" y="83" width="300" height="54" rx="27" fill="{p['teal_bg']}"/><text x="2028" y="117" font-size="19" font-weight="700" fill="{teal}">Local vision model</text>
<rect x="65" y="260" width="2270" height="825" rx="35" fill="{p['panel']}" stroke="{p['border']}" stroke-width="2"/>
<text x="105" y="302" font-size="17" letter-spacing="2" font-weight="750" fill="{p['muted']}">QUESTION → EVIDENCE → ANALYSIS → ANSWER</text>
''']

    def text(x, y, value, size=20, color=None, weight=400):
        s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color or p["muted"]}">{escape(value)}</text>')

    def icon(kind, x, y, color):
        s.append(f'<g transform="translate({x},{y})" fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">')
        paths = {
            "chat": '<rect x="3" y="3" width="38" height="30" rx="7"/><path d="M14 33v8l10-8"/><path d="M13 18h.1M22 18h.1M31 18h.1" stroke-width="5"/>',
            "agent": '<rect x="7" y="8" width="31" height="28" rx="7"/><path d="M22 2v6M16 23h.1M29 23h.1M16 30h13M2 20h5M38 20h5"/>',
            "tools": '<path d="M27 5a11 11 0 0 0-12 14L4 30a6 6 0 0 0 9 9l11-11A11 11 0 0 0 38 16l-7 7-7-2-2-7z"/>',
            "archive": '<ellipse cx="22" cy="8" rx="16" ry="6"/><path d="M6 8v24c0 4 7 7 16 7s16-3 16-7V8M6 20c0 4 7 7 16 7s16-3 16-7"/>',
            "shield": '<path d="M22 3 38 9v11c0 10-7 16-16 20C13 36 6 30 6 20V9z"/><path d="m15 21 5 5 9-10"/>',
            "model": '<rect x="5" y="4" width="34" height="10" rx="4"/><rect x="5" y="18" width="34" height="10" rx="4"/><rect x="5" y="32" width="34" height="10" rx="4"/><path d="M12 9h.1M12 23h.1M12 37h.1"/>',
            "vision": '<rect x="3" y="5" width="38" height="34" rx="5"/><path d="M8 29l6-7 5 3 5-13 5 16 5-7 3 3M9 12h.1"/>',
            "media": '<rect x="4" y="5" width="36" height="34" rx="5"/><path d="m8 33 10-11 7 6 5-5 6 10M15 15h.1" stroke-width="3.2"/>',
        }
        s.append(paths[kind] + '</g>')

    def card(x, y, title, subtitle, lines, kind, accent="purple", badge=None):
        color = purple if accent == "purple" else teal
        icon_bg = p["purple_bg"] if accent == "purple" else p["teal_bg"]
        s.append(f'<rect x="{x}" y="{y}" width="430" height="250" rx="23" fill="{p["card"]}" stroke="{p["border"]}" stroke-width="2" filter="url(#shadow)"/>')
        s.append(f'<rect x="{x+24}" y="{y+24}" width="65" height="65" rx="17" fill="{icon_bg}"/>')
        icon(kind, x+34, y+34, color)
        if badge:
            s.append(f'<rect x="{x+300}" y="{y+35}" width="106" height="34" rx="17" fill="{icon_bg}"/>')
            text(x+318, y+57, badge, 14, color, 750)
        text(x+24, y+126, title, 27, p["ink"], 750)
        text(x+24, y+157, subtitle, 18, color, 650)
        for i, line in enumerate(lines):
            text(x+24, y+190+i*27, line, 17)

    # Forward and return lanes make request/response direction explicit.
    def hpair(left, right, upper, lower):
        s.append(f'<path d="M{left} 414 H{right-8}" fill="none" stroke="{purple}" stroke-width="3" marker-end="url(#arrow)"/>')
        s.append(f'<path d="M{right} 480 H{left+8}" fill="none" stroke="{p["line"]}" stroke-width="2.5" marker-end="url(#arrow)"/>')
        text((left+right)/2, 400, upper, 14, purple, 700)
        text((left+right)/2, 506, lower, 14, p["muted"], 650)

    # Text labels are centered in the space between the cards.
    s.append('<g text-anchor="middle">')
    hpair(540, 675, "QUESTION", "ANSWER")
    hpair(1105, 1240, "TOOL CALL", "RESULTS")
    hpair(1670, 1805, "QUERY", "DATA")
    s.append('</g>')
    s.append(f'<path d="M850 590 V690" stroke="{purple}" stroke-width="3" fill="none" marker-end="url(#arrow)"/>')
    s.append(f'<path d="M915 690 V590" stroke="{p["line"]}" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>')
    s.append(f'<path d="M1415 590 V690" stroke="{teal}" stroke-width="3" fill="none" marker-end="url(#arrow-teal)"/>')
    s.append(f'<path d="M1480 690 V590" stroke="{p["line"]}" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>')
    s.append(f'<path d="M1670 816 H1795" stroke="{teal}" stroke-width="3" fill="none" marker-end="url(#arrow-teal)"/>')
    s.append(f'<path d="M2020 940 V1028 H92 V543 H100" stroke="{p["line"]}" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>')
    text(881, 645, "PROMPT / DECISION", 14, p["muted"], 700)
    text(1438, 645, "IMAGE / RESULTS", 14, p["muted"], 700)
    text(1738, 798, "SAVE", 14, teal, 700)
    text(920, 1055, "LOCAL MEDIA DISPLAYED IN THE CHAT", 15, p["muted"], 700)

    card(110, 340, "Gradio workspace", "QUESTION & PRESENTATION", ["Chat, images and audio playback", "Request monitor · session history"], "chat")
    card(675, 340, "Agent orchestration", "PYDANTIC AI + TURN SERVICE", ["Plans requests and validates calls", "Produces grounded responses"], "agent")
    card(1240, 340, "BRAMS & science tools", "PYTHON + PYBRAMS", ["Stations, geography, availability", "Recordings, spectrograms and PSD"], "tools")
    card(1805, 340, "BRAMS archive", "EXTERNAL DATA SERVICE", ["Receiver and beacon metadata", "Archived PNGs and WAV recordings"], "archive")
    card(110, 690, "Evidence safeguards", "APPLIED ACROSS THE TURN", ["Grounding and coverage checks", "Limits, retries and recovery"], "shield")
    card(675, 690, "Language model", "SELECTABLE ENDPOINT", ["Local llama.cpp or hosted API", "Tool decisions and explanations"], "model")
    card(1240, 690, "Meteor detection", "LOCAL VISION MODEL", ["YOLOv11-n + ONNX Runtime", "Candidate boxes and estimated times"], "vision", "teal", "LOCAL")
    card(1805, 690, "Local media", "ARTIFACTS + PROVENANCE", ["Original PNG/WAV and audio preview", "Annotated images and checksums"], "media", "teal")
    s.append(f'<rect x="80" y="1138" width="2240" height="143" rx="24" fill="{p["card"]}" stroke="{p["border"]}" stroke-width="2"/>')
    text(115, 1182, "A COMPLETE RESEARCH LOOP", 16, purple, 750)
    text(115, 1224, "Ask a question", 25, p["ink"], 750)
    text(490, 1224, "Retrieve verified data", 25, p["ink"], 750)
    text(1040, 1224, "Detect candidate echoes", 25, p["ink"], 750)
    text(1660, 1224, "Review results & images", 25, p["ink"], 750)
    text(115, 1255, "Relative time and regional queries", 17)
    text(490, 1255, "Coverage and source references", 17)
    text(1040, 1255, "Counts, confidence and coverage", 17)
    text(1660, 1255, "Image bytes remain local", 17)
    s.append('</g></svg>')
    return "\n".join(s)


if __name__ == "__main__":
    for dark in (False, True):
        suffix = "-dark" if dark else ""
        svg = HERE / f"readme-architecture{suffix}.svg"
        png = HERE / f"readme-architecture{suffix}.png"
        svg.write_text(render(dark), encoding="utf-8")
        subprocess.run(["rsvg-convert", "-w", "2400", "-o", str(png), str(svg)], check=True)
