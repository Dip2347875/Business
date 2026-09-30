#!/usr/bin/env python3
"""RSS -> AI (OpenRouter/Cohere fallback) -> evidence-based SEO article -> Blogger."""
import os, re, sys, json, time, random, hashlib
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from urllib.parse import quote

import requests
import feedparser
from bs4 import BeautifulSoup

CATEGORIES = ["Business News", "Startups", "Finance and Banking", "Stock Market",
              "Corporate News", "General Economy", "Business Idea"]

FEEDS = [
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
    ("Google News Business", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"),
    ("NBC News", "https://feeds.nbcnews.com/nbcnews/public/news"),
    ("The New York Times", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
]

KEYWORDS = {
    "Business News": ["business", "company", "companies", "industry", "retail", "trade", "tariff", "consumer",
                      "sales", "supply chain", "manufactur", "jobs", "workers", "market share", "merger"],
    "Startups": ["startup", "start-up", "founder", "venture", "funding", "seed round", "series a", "series b",
                 "unicorn", "incubator", "ai company", "entrepreneur", "raises $", "valuation"],
    "Finance and Banking": ["bank", "banking", "loan", "credit", "mortgage", "interest rate", "fed ", "federal reserve",
                            "lender", "deposit", "fintech", "payment", "debt", "central bank", "insurance"],
    "Stock Market": ["stock", "shares", "wall street", "nasdaq", "dow", "s&p", "investor", "earnings", "rally",
                     "sell-off", "ftse", "index", "ipo", "bond", "crypto", "bitcoin"],
    "Corporate News": ["ceo", "chief executive", "layoff", "acquisition", "acquire", "takeover", "profit", "revenue",
                       "quarterly", "board", "resign", "lawsuit", "antitrust", "results", "boss"],
    "General Economy": ["economy", "economic", "inflation", "gdp", "recession", "unemployment", "growth", "prices",
                        "cost of living", "budget", "tax", "imf", "world bank", "oil", "energy prices", "currency"],
}

OR_MODELS = [m for m in os.getenv("OPENROUTER_MODELS", "meta-llama/llama-3.3-70b-instruct:free,"
             "google/gemini-2.0-flash-exp:free,deepseek/deepseek-chat-v3-0324:free").split(",") if m]
CO_MODELS = [m for m in os.getenv("COHERE_MODELS", "command-a-03-2025,command-r-plus-08-2024").split(",") if m]

STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BlogBot/1.0)"}


class Retry(Exception):
    """Expected failure: nothing is posted, state is untouched, next run tries again."""


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ state
def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {}
    s.setdefault("cat_index", 0)
    s.setdefault("next_due", None)
    s.setdefault("seen", [])
    s.setdefault("titles", [])
    return s


def save_state(s):
    s["seen"] = s["seen"][-1000:]
    s["titles"] = s["titles"][-400:]
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=1)


# ------------------------------------------------------------------ RSS
def clean(t):
    return re.sub(r"\s+", " ", BeautifulSoup(t or "", "html.parser").get_text(" ", strip=True)).strip()


def hid(link):
    return hashlib.sha1(link.split("?")[0].encode()).hexdigest()[:16]


def upgrade_img(u):
    return re.sub(r"/(ace/standard|news)/\d+/", r"/\1/976/", u)


def entry_image(e):
    c = []
    for k in ("media_content", "media_thumbnail"):
        for m in e.get(k, []) or []:
            if m.get("url"):
                try:
                    w = int(m.get("width") or 0)
                except ValueError:
                    w = 0
                c.append((w, m["url"]))
    for enc in e.get("enclosures", []) or []:
        if str(enc.get("type", "")).startswith("image") and enc.get("href"):
            c.append((1, enc["href"]))
    for l in e.get("links", []) or []:
        if str(l.get("type", "")).startswith("image") and l.get("href"):
            c.append((1, l["href"]))
    img = BeautifulSoup(e.get("summary", "") or "", "html.parser").find("img")
    if img and img.get("src"):
        c.append((0, img["src"]))
    if not c:
        return None
    c.sort(key=lambda x: x[0], reverse=True)
    return upgrade_img(c[0][1])


def collect():
    out = []
    for name, url in FEEDS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            feed = feedparser.parse(r.content)
        except Exception as ex:
            log("feed fail", name, ex)
            continue
        for e in feed.entries[:40]:
            title = clean(e.get("title", ""))
            link = e.get("link", "")
            if not title or not link:
                continue
            if "Google" in name:
                title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title)
            out.append({"title": title, "link": link, "source": name,
                        "summary": clean(e.get("summary", ""))[:1200], "image": entry_image(e)})
    log(f"collected {len(out)} entries")
    return out


def score(e, cat):
    t = f"{e['title']} {e['summary']}".lower()
    return sum(1 for k in KEYWORDS[cat] if k in t)


def best_score(e):
    return max(score(e, c) for c in KEYWORDS)


def is_dup(e, state):
    if hid(e["link"]) in state["seen"]:
        return True
    return any(SequenceMatcher(None, e["title"].lower(), t.lower()).ratio() > 0.75 for t in state["titles"])


def ranked(cat, entries, state):
    fresh = [e for e in entries if not is_dup(e, state)]
    keyf = best_score if cat == "Business Idea" else (lambda e: score(e, cat))
    pool = sorted([e for e in fresh if keyf(e) > 0], key=lambda e: (keyf(e), bool(e["image"])), reverse=True)
    if not pool:
        pool = sorted([e for e in fresh if best_score(e) > 0],
                      key=lambda e: (best_score(e), bool(e["image"])), reverse=True)
    top = pool[:6]
    random.shuffle(top)
    return top + pool[6:]


def words(t):
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 3}


def related(e, entries):
    ew = words(e["title"])
    out = []
    for o in entries:
        if o["link"] != e["link"] and len(ew & words(o["title"])) >= 3:
            out.append(o)
    return out[:3]


def article_text(e):
    if "news.google.com" in e["link"]:
        return e["summary"]
    try:
        r = requests.get(e["link"], headers=HEADERS, timeout=25)
        soup = BeautifulSoup(r.text, "html.parser")
        if not e["image"]:
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                e["image"] = og["content"]
        ps = [clean(p.get_text()) for p in soup.find_all("p")]
        txt = " ".join(p for p in ps if len(p) > 60)[:5000]
        return txt or e["summary"]
    except Exception:
        return e["summary"]


# ------------------------------------------------------------------ images
def valid_image(u, timeout=60):
    try:
        r = requests.get(u, headers=HEADERS, timeout=timeout)
        return (r.status_code == 200 and r.headers.get("content-type", "").startswith("image/")
                and len(r.content) > 5000)
    except Exception:
        return False


def generate_image(prompt):
    url = (f"https://image.pollinations.ai/prompt/"
           f"{quote(prompt[:300] + ', editorial photo, realistic, high quality, no text')}"
           f"?width=1200&height=630&nologo=true&seed={random.randint(1, 99999)}")
    for _ in range(2):
        if valid_image(url, 90):
            return url
        time.sleep(5)
    return None
  # ------------------------------------------------------------------ AI
SYSTEM = ("You are a senior business journalist, fact-checker and SEO editor. You write original, human-sounding "
          "English articles that analyse facts and evidence. You never invent facts, numbers or quotes. "
          "You reply with valid JSON only.")


def build_prompt(cat, e=None, text="", rel=None, avoid=None):
    idea = ""
    if cat == "Business Idea":
        idea = ("This is a BUSINESS IDEA article: turn the trend into a practical idea - the opportunity, target "
                "customer, how to start, estimated startup cost range (clearly marked as estimate), revenue model, "
                "risks, and a 30-day action plan. ")
    if e:
        rel_txt = "\n".join(f"- {o['source']}: {o['title']} - {o['summary'][:300]}" for o in (rel or [])) or "(none)"
        src = f"""Write a complete blog post for the category "{cat}" analysing this news story.

MAIN REPORT ({e['source']}): {e['title']}
FACTS / CONTEXT:
{text}

OTHER REPORTS ON THE SAME STORY (use as cross-checking evidence):
{rel_txt}

Evidence rules: use ONLY facts found above. Attribute every key claim to its source (e.g. "according to BBC Business").
Where the reports agree, say so; where details differ or are missing, say so. Include a section that separates
"What is confirmed" from "What is still unclear". Never invent numbers, dates, quotes or named experts."""
    else:
        src = f"""No live news feed is available right now. Write a complete, fact-based analytical article for the
category "{cat}" on a topic of ongoing relevance (an explainer/analysis, NOT breaking news).
Pick a specific, useful topic. Do NOT claim anything happened "today" or "recently", do not invent events, quotes,
named experts or precise statistics. State only well-established facts you are highly confident about, and label any
figure as approximate. Include a section on evidence: what the data and history generally show, and its limits.
Do NOT reuse or closely resemble these already-published topics: {json.dumps((avoid or [])[-40:])}"""
    return f"""{src}

{idea}Requirements:
- 1000-1400 words, simple language, short sentences, smooth transitions, easy to read on mobile.
- Structure: gripping Hook, Curiosity gap, Context/Background, Main Story, Key Facts (bullet list), Timeline where
  relevant, Evidence, Analysis, Expert-style perspective (no invented quotes), Real-world Human Impact, one Surprising
  detail, Suspense/build-up between sections, Key Takeaways (bullets), Powerful Conclusion, short Call to Action.
- SEO: one focus keyword in the title, first paragraph, one H2 and the conclusion; natural variations; keyword-rich
  but natural H2/H3 headings; no keyword stuffing.
- HTML only in "html": <h2>, <h3>, <p>, <ul><li>, <strong>, <blockquote>. No <h1>, <img>, <script>.
- Title under 60 characters, click-worthy and accurate.

Return ONLY this JSON:
{{"title": "...", "meta_description": "<=155 chars", "focus_keyword": "...", "labels": ["up to 3 short tags"],
 "image_prompt": "short vivid photo description, no text or logos",
 "html": "<p>...</p>...", "faq": [{{"q": "...", "a": "..."}}, {{"q": "...", "a": "..."}}, {{"q": "...", "a": "..."}}]}}"""


def call_openrouter(prompt, model):
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                               "Content-Type": "application/json",
                               "HTTP-Referer": "https://github.com", "X-Title": "Auto Blog"},
                      json={"model": model, "max_tokens": 4500, "temperature": 0.6,
                            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
                      timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_cohere(prompt, model):
    r = requests.post("https://api.cohere.com/v2/chat",
                      headers={"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}",
                               "Content-Type": "application/json"},
                      json={"model": model, "max_tokens": 4500, "temperature": 0.6,
                            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
                      timeout=180)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["message"]["content"])


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("no json")
    return json.loads(t[a:b + 1], strict=False)


def sanitize(html_):
    soup = BeautifulSoup(html_, "html.parser")
    for t in soup.find_all(["script", "style", "iframe", "h1", "img"]):
        t.decompose()
    return str(soup)


def generate(prompt, used_titles):
    plan = []
    for i in range(max(len(OR_MODELS), len(CO_MODELS))):
        if i < len(OR_MODELS) and os.getenv("OPENROUTER_API_KEY"):
            plan.append(("openrouter", OR_MODELS[i]))
        if i < len(CO_MODELS) and os.getenv("COHERE_API_KEY"):
            plan.append(("cohere", CO_MODELS[i]))
    if not plan:
        raise RuntimeError("No AI API keys set")
    for _ in range(2):
        for prov, model in plan:
            try:
                log(f"AI try: {prov} / {model}")
                raw = call_openrouter(prompt, model) if prov == "openrouter" else call_cohere(prompt, model)
                art = parse_json(raw)
                art["html"] = sanitize(art["html"])
                n = len(BeautifulSoup(art["html"], "html.parser").get_text().split())
                if not art.get("title") or n < 500:
                    raise ValueError(f"too short ({n} words)")
                if any(SequenceMatcher(None, art["title"].lower(), t.lower()).ratio() > 0.75 for t in used_titles):
                    raise ValueError("title too similar to an earlier post")
                return art
            except Exception as ex:
                log("AI fail:", prov, model, str(ex)[:200])
                time.sleep(3)
        time.sleep(20)
    raise Retry("all AI providers failed")


# ------------------------------------------------------------------ Blogger
def blogger_token():
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": os.environ["GOOGLE_CLIENT_ID"], "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_REFRESH_TOKEN"], "grant_type": "refresh_token"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def blog_info(token):
    try:
        r = requests.get(f"https://www.googleapis.com/blogger/v3/blogs/{os.environ['BLOGGER_BLOG_ID']}",
                         headers={"Authorization": f"Bearer {token}"}, timeout=30)
        j = r.json()
        url = j.get("url", "")
        return j.get("name") or "News Desk", (url if url.endswith("/") else url + "/") if url else ""
    except Exception:
        return "News Desk", ""


def build_content(art, e, cat, img, credit, blog_name, blog_url):
    title = art["title"].strip()
    desc = (art.get("meta_description") or "")[:155]
    faq = [f for f in art.get("faq", []) if f.get("q") and f.get("a")][:5]
    faq_html = ""
    if faq:
        faq_html = "<h2>Frequently Asked Questions</h2>" + "".join(
            f"<h3>{f['q']}</h3><p>{f['a']}</p>" for f in faq)
    now = datetime.now(timezone.utc).isoformat()
    graph = [{"@type": "NewsArticle", "headline": title, "description": desc, "image": [img],
              "datePublished": now, "dateModified": now, "articleSection": cat,
              "author": {"@type": "Organization", "name": blog_name},
              "publisher": {"@type": "Organization", "name": blog_name}}]
    if faq:
        graph.append({"@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": f["q"],
             "acceptedAnswer": {"@type": "Answer", "text": f["a"]}} for f in faq]})
    ld = json.dumps({"@context": "https://schema.org", "@graph": graph}).replace("</", "<\\/")
    disclaimer = ""
    if cat in ("Finance and Banking", "Stock Market", "General Economy", "Business Idea"):
        disclaimer = ("<p><em>Disclaimer: This article is for informational purposes only and is not financial "
                      "or investment advice. Always do your own research.</em></p>")
    ref = ""
    if e.get("link"):
        ref = (f'<p><small>Reference: <a href="{e["link"]}" target="_blank" rel="nofollow noopener">'
               f'{e["source"]}</a>. This article is our own analysis based on public reporting.</small></p>')
    more = ""
    if blog_url:
        more = f'<p>More from <a href="{blog_url}search/label/{quote(cat)}">{cat}</a></p>'
    alt = title.replace('"', "'")
    return (f'<div class="separator" style="text-align:center;"><img src="{img}" alt="{alt}" title="{alt}" '
            f'width="1200" style="max-width:100%;height:auto;" /></div>'
            f'<p style="text-align:center;"><small>Image: {credit}</small></p>'
            f'{art["html"]}{faq_html}{disclaimer}{ref}{more}'
            f'<script type="application/ld+json">{ld}</script>')


def publish(art, content, cat, token):
    labels = [cat] + [l.strip() for l in art.get("labels", [])[:3] if l and l.strip() and l.strip() != cat]
    r = requests.post(
        f"https://www.googleapis.com/blogger/v3/blogs/{os.environ['BLOGGER_BLOG_ID']}/posts?isDraft=false",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"kind": "blogger#post", "title": art["title"].strip(), "content": content, "labels": labels},
        timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"Blogger error {r.status_code}: {r.text[:300]}")
    return r.json().get("url")


# ------------------------------------------------------------------ main
def run():
    state = load_state()
    now = datetime.now(timezone.utc)
    force = os.getenv("FORCE", "0") in ("1", "true", "yes")
    if not force and state["next_due"] and now < datetime.fromisoformat(state["next_due"]):
        log("not due until", state["next_due"])
        return

    cat = CATEGORIES[state["cat_index"] % len(CATEGORIES)]
    log("category:", cat)
    token = blogger_token()
    blog_name, blog_url = blog_info(token)

    entries = collect()
    cands = ranked(cat, entries, state) if entries else []
    e = art = img = credit = None

    if cands:
        gen_ok = True
        for cand in cands[:8]:
            text = article_text(cand)
            if cand.get("image") and valid_image(cand["image"], 25):
                img, credit = cand["image"], cand["source"]
            elif gen_ok:
                img = generate_image(cand["title"])
                credit = "AI-generated illustration"
                if not img:
                    gen_ok = False
            else:
                img = None
            if img:
                e = cand
                log("picked:", e["title"], "|", e["source"])
                prompt = build_prompt(cat, e, text, related(e, entries))
                art = generate(prompt, state["titles"])
                break
        if not e:
            raise Retry("no usable image for any candidate - will retry next run")
    else:
        log("no usable RSS entries -> AI writes the article itself")
        art = generate(build_prompt(cat, avoid=state["titles"]), state["titles"])
        img = generate_image(art.get("image_prompt") or art["title"])
        credit = "AI-generated illustration"
        if not img:
            raise Retry("could not create an image - will retry next run")
        e = {"title": art["title"], "link": None, "source": "Editorial"}

    content = build_content(art, e, cat, img, credit, blog_name, blog_url)
    url = publish(art, content, cat, token)
    log("POSTED:", url)

    if e.get("link"):
        state["seen"].append(hid(e["link"]))
    state["titles"] += [e["title"], art["title"]]
    state["cat_index"] = (state["cat_index"] + 1) % len(CATEGORIES)
    state["next_due"] = (datetime.now(timezone.utc) + timedelta(minutes=random.randint(50, 70))).isoformat()
    save_state(state)


if __name__ == "__main__":
    try:
        run()
    except Retry as ex:
        log("RETRY LATER:", ex)   # exit 0: nothing posted, timer unchanged, next 10-min run tries again
    except Exception as ex:
        log("FAILED:", ex)
        sys.exit(1)
