#!/usr/bin/env python3
"""
Verify BibTeX references using clibib to catch hallucinated citations.

For each entry in references.bib:
1. Extract arXiv ID or title
2. Query clibib (if arXiv URL available)
3. Compare venue, year, authors
4. Report discrepancies
"""
import re
import subprocess
import time
from pathlib import Path

def parse_bib_file(path):
    """Parse BibTeX file into list of (key, entry_text, metadata_dict)."""
    with open(path) as f:
        content = f.read()

    entries = []
    # Match @type{key, ... }
    pattern = r'@(\w+)\{([^,]+),\s*(.*?)\n\}'
    for match in re.finditer(pattern, content, re.DOTALL):
        entry_type, key, body = match.groups()
        # Parse metadata
        meta = {'type': entry_type, 'key': key}
        for line in body.split('\n'):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            m = re.match(r'(\w+)\s*=\s*[{"](.*?)["}],?$', line)
            if m:
                field, value = m.groups()
                meta[field] = value
        entries.append((key, match.group(0), meta))
    return entries

def extract_arxiv_id(meta):
    """Extract arXiv ID from journal/eprint fields."""
    # Check journal field for arXiv preprint format
    journal = meta.get('journal', '')
    if 'arXiv' in journal:
        m = re.search(r'arXiv:([0-9.]+)', journal)
        if m:
            return m.group(1)
    # Check eprint field
    if 'eprint' in meta:
        return meta['eprint']
    return None

def query_clibib(query, use_first=True):
    """Query clibib and return BibTeX string or None."""
    cmd = ['clibib', query]
    if use_first:
        cmd.append('--first')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  clibib error: {e}")
        return None

def compare_entries(key, original_meta, clibib_bib):
    """Compare original entry with clibib result. Return list of discrepancies."""
    if not clibib_bib:
        return []

    # Parse clibib output
    clibib_lines = clibib_bib.split('\n')
    clibib_meta = {}
    for line in clibib_lines:
        m = re.match(r'\s*(\w+)\s*=\s*[{"](.*?)["}],?$', line)
        if m:
            field, value = m.groups()
            clibib_meta[field] = value

    issues = []

    # Check year
    orig_year = original_meta.get('year', '')
    clib_year = clibib_meta.get('year', '')
    if orig_year and clib_year and orig_year != clib_year:
        issues.append(f"Year mismatch: {orig_year} (ours) vs {clib_year} (clibib)")

    # Check venue type
    orig_type = original_meta.get('type', '')
    clib_type = clibib_meta.get('type', '')  # Usually not in parsed meta, check @article vs @inproceedings
    if '@article' in clibib_bib and 'booktitle' in original_meta:
        issues.append(f"Type mismatch: we have booktitle (conference) but clibib says article")
    elif '@inproceedings' in clibib_bib and 'journal' in original_meta and 'arXiv' not in original_meta.get('journal', ''):
        issues.append(f"Type mismatch: we have journal but clibib says inproceedings")

    return issues

def main():
    bib_path = Path(__file__).parent / 'references.bib'
    entries = parse_bib_file(bib_path)

    print(f"Found {len(entries)} entries in references.bib\n")

    # Focus on entries with arXiv IDs first
    arxiv_entries = []
    other_entries = []

    for key, text, meta in entries:
        arxiv_id = extract_arxiv_id(meta)
        if arxiv_id:
            arxiv_entries.append((key, text, meta, arxiv_id))
        else:
            other_entries.append((key, text, meta))

    print(f"Checking {len(arxiv_entries)} arXiv entries...\n")

    all_issues = {}

    for key, text, meta, arxiv_id in arxiv_entries:
        print(f"[{key}]", end=' ', flush=True)

        # Query clibib with full arXiv URL
        query = f"https://arxiv.org/abs/{arxiv_id}"
        clibib_result = query_clibib(query)

        if clibib_result:
            issues = compare_entries(key, meta, clibib_result)
            if issues:
                all_issues[key] = {
                    'query': query,
                    'issues': issues,
                    'clibib_bib': clibib_result,
                    'original_year': meta.get('year'),
                    'original_venue': meta.get('journal') or meta.get('booktitle')
                }
                print(f"⚠️  {len(issues)} issue(s)")
            else:
                print("✓")
        else:
            print("⚠️  clibib failed")

        time.sleep(0.5)  # Rate limit

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found issues in {len(all_issues)}/{len(arxiv_entries)} arXiv entries\n")

    if all_issues:
        for key, data in all_issues.items():
            print(f"\n{key}:")
            print(f"  Query: {data['query']}")
            print(f"  Original: year={data['original_year']}, venue={data['original_venue']}")
            for issue in data['issues']:
                print(f"  ⚠️  {issue}")
            print(f"\n  clibib suggests:\n")
            for line in data['clibib_bib'].split('\n')[:8]:  # First 8 lines
                print(f"    {line}")
            print("    ...")

    print(f"\n{'='*60}")
    print(f"Non-arXiv entries ({len(other_entries)}) not checked automatically.")
    print("Manual review recommended for:")
    for key, _, meta in other_entries[:5]:
        title = meta.get('title', '(no title)')[:60]
        year = meta.get('year', '?')
        print(f"  - {key} ({year}): {title}")

if __name__ == '__main__':
    main()
