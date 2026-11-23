"""Find literature and check DOI metadata with Crossref and Europe PMC."""

import functools
import html
import re
import urllib.parse

import requests


def _clean_doi(value):
    doi = re.sub(r"^(?:DOI:\s*|https?://(?:dx\.)?doi\.org/)", "", (value or "").strip(), flags=re.I)
    return doi if re.fullmatch(r"10\.\d{4,9}/[^\s<>\"']{1,220}", doi) else ""


def _literature_snippet(text, limit=600):
    return " ".join(re.sub(r"<[^>]*>", " ", html.unescape(text or "")).split())[:limit]


def cbyd_precedent_literature(precedent):
    """Preserve CbyD's curated DOI/citation attribution without live network verification."""
    doi = _clean_doi(precedent.get("doi") or "")
    if not doi:
        return None
    return {
        "doi": doi,
        "title": _literature_snippet(precedent.get("citation") or "Chemistry by Design precedent"),
        "snippet": _literature_snippet(precedent.get("citation")),
        "source": "Chemistry by Design (CC BY 4.0)",
    }


@functools.lru_cache(maxsize=256)
def _verify_doi_metadata(value):
    doi = _clean_doi(value)
    if not doi:
        return None
    try:
        encoded = urllib.parse.quote(doi, safe="/")
        result = requests.get(f"https://api.crossref.org/works/{encoded}", timeout=3)
        result.raise_for_status()
        work = result.json()["message"]
        verified = _clean_doi(work.get("DOI"))
        if verified.lower() == doi.lower():
            return {"doi": verified, "title": _literature_snippet((work.get("title") or [""])[0]),
                    "snippet": _literature_snippet(work.get("abstract"))}
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


@functools.lru_cache(maxsize=128)
def fetch_verified_literature_dois(product_smiles, reaction_name):
    """Use product PubMed cross-references and real transformation keywords to find verified DOI candidates."""
    pmids = []
    if product_smiles:
        try:
            pubchem = requests.post("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/xrefs/PubMedID/JSON",
                                    data={"smiles": product_smiles}, timeout=3)
            pubchem.raise_for_status()
            rows = pubchem.json()["InformationList"]["Information"]
            pmids = list(dict.fromkeys(str(pmid) for row in rows for pmid in row.get("PubMedID", []) if str(pmid).isdigit()))[:8]
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError):
            pass
    terms = [re.sub(r"[^\w -]", " ", term).strip()[:100] for term in re.split(r"\s+OR\s+|\s+/\s+", reaction_name or "", flags=re.I)]
    filters = [f'TITLE_ABS:"{term}"' for term in terms[:4] if term and term.lower() != "synthetic transformation"]
    filters.extend(f"(EXT_ID:{pmid} AND SRC:MED)" for pmid in pmids)
    if not filters:
        return ()
    try:
        response = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                                params={"query": " OR ".join(filters), "format": "json", "resultType": "core", "pageSize": 12}, timeout=3)
        response.raise_for_status()
        articles = response.json().get("resultList", {}).get("result", [])
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return ()
    verified, seen = [], set()
    articles = sorted(articles, key=lambda a: str(a.get("pmid")) not in pmids)
    for article in [a for a in articles if _clean_doi(a.get("doi"))][:6]:
        metadata = _verify_doi_metadata(article["doi"])
        if metadata and metadata["doi"].lower() not in seen:
            seen.add(metadata["doi"].lower())
            verified.append(dict(metadata, snippet=metadata["snippet"] or _literature_snippet(article.get("abstractText")),
                                 product_pubmed_xref=str(article.get("pmid")) in pmids))
        if len(verified) >= 3:
            break
    return tuple(verified)
