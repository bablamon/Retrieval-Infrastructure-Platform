import hashlib
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid",
    "_ga", "_gl", "igshid", "s_cid", "ncid",
})


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Clean path: strip trailing slash except for root
    path = parsed.path.rstrip("/") or "/"

    # Filter and sort query params
    qs = parse_qs(parsed.query, keep_blank_values=False)
    filtered_qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
    sorted_query = urlencode(sorted(filtered_qs.items()), doseq=True)

    return urlunparse((scheme, netloc, path, parsed.params, sorted_query, ""))


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_same_domain(url1: str, url2: str) -> bool:
    return extract_domain(url1) == extract_domain(url2)


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def url_to_key(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


def deduplicate_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        try:
            key = url_to_key(url)
            if key not in seen:
                seen.add(key)
                result.append(url)
        except Exception:
            continue
    return result
