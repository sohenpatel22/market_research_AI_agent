"""Minimal SEC EDGAR client: ticker -> CIK lookup"""

import json
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from market_research_agent.config import settings
from market_research_agent.data.cache import raw_path, read_cached, write_cache

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

REQUEST_DELAY_SECONDS = 0.3


def _headers() -> dict[str, str]:
    return {"User-Agent": settings.sec_edgar_user_agent}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def _get(url: str) -> requests.Response:
    response = requests.get(url, headers=_headers(), timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response


def get_cik_for_ticker(ticker: str) -> str:
    """Resolve a ticker to its zero-padded 10-digit CIK, using a cached lookup table"""
    cached = read_cached("edgar", "company_tickers.json")
    if cached is None:
        cached = _get(COMPANY_TICKERS_URL).text
        write_cache(cached, "edgar", "company_tickers.json")

    for entry in json.loads(cached).values():
        if entry["ticker"].upper() == ticker.upper():
            return f"{entry['cik_str']:0>10}"
    raise ValueError(f"No CIK found for ticker {ticker!r}")


@dataclass
class FilingRef:
    ticker: str
    cik: str
    accession_number: str
    filing_type: str
    filed_date: str
    primary_document: str


def get_recent_filings(
    ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit_per_form: int = 1
) -> list[FilingRef]:
    """List the most recent filings of the given forms for a ticker"""
    cik = get_cik_for_ticker(ticker)

    cache_key = ("edgar", f"submissions_{ticker.upper()}.json")
    cached = read_cached(*cache_key)
    if cached is None:
        cached = _get(SUBMISSIONS_URL.format(cik=cik)).text
        write_cache(cached, *cache_key)

    recent = json.loads(cached)["filings"]["recent"]

    counts: dict[str, int] = dict.fromkeys(forms, 0)
    filings: list[FilingRef] = []
    for form, accession, filed_date, primary_doc in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["filingDate"],
        recent["primaryDocument"],
        strict=True,
    ):
        if form not in forms or counts[form] >= limit_per_form:
            continue
        counts[form] += 1
        filings.append(
            FilingRef(
                ticker=ticker.upper(),
                cik=cik,
                accession_number=accession,
                filing_type=form,
                filed_date=filed_date,
                primary_document=primary_doc,
            )
        )
    return filings


def fetch_filing_text(filing: FilingRef) -> str:
    """Download (or load from cache) a filing document and extract plain text"""
    cache_file = raw_path("filings", filing.ticker, f"{filing.accession_number}.html")
    if cache_file.exists():
        html = cache_file.read_text(encoding="utf-8")
    else:
        url = ARCHIVES_URL.format(
            cik=int(filing.cik),
            accession_nodash=filing.accession_number.replace("-", ""),
            document=filing.primary_document,
        )
        html = _get(url).text
        cache_file.write_text(html, encoding="utf-8")

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")
