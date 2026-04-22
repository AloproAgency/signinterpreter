"""
One-shot migration to deduplicate Word rows that differ only by case.

Problem
-------
On case-insensitive filesystems (default macOS APFS) `data/templates/bon`
and `data/templates/BON` map to the SAME inode, but the SQLite DB stored
each as a distinct Word row (case-sensitive comparison).  All
contributions to the upper-case row claimed templates that the lower-
case row also reported, and the two `template_count` columns drifted as
the API only synced the lowercase form.

This script:

1. Groups all Word rows by `normalize_word(name)`.
2. For each group with >1 row, picks a CANONICAL row (lowercase
   version, or the first one if no lowercase exists).
3. Re-targets every Contribution.word_id from the duplicates to the
   canonical row.
4. Re-targets every DailyAssignment.word_id from the duplicates to the
   canonical row (foreign key into Word).
5. Renames the canonical row's name to its normalized form (in case
   the canonical was 'Bon ' instead of 'bon').
6. Deletes the now-orphan duplicate Word rows.
7. Recomputes template_count for every Word from the on-disk template
   directory.
8. Renames any data/templates/<NAME>/ folder to its normalized form.
   On case-insensitive filesystems both names point to the same inode
   so this is a no-op metadata change; on case-sensitive filesystems
   it merges the two physical directories.

Run idempotently; safe to re-run.
"""
from __future__ import annotations

import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sqlalchemy import or_

from ml.constants import TEMPLATE_DIR
from ml.template_manager import normalize_word
from server.database import (
    SessionLocal, Word, Contribution, DailyAssignment,
)


def _pick_canonical(rows: List[Word]) -> Word:
    """Among duplicate rows for the same normalized name, prefer the
    one whose stored name already equals the normalized form, then
    the one with the most contributions (so we don't have to move
    too many FK rows), then the lowest id."""
    rows_lower = [r for r in rows if r.name == normalize_word(r.name)]
    if rows_lower:
        rows = rows_lower
    rows.sort(key=lambda r: (-len(r.contributions or []), r.id))
    return rows[0]


def _merge_words(db) -> Dict[str, int]:
    """Merge case-duplicate Word rows.  Returns a dict {merged: lost_id}."""
    by_norm: Dict[str, List[Word]] = defaultdict(list)
    for w in db.query(Word).all():
        by_norm[normalize_word(w.name)].append(w)

    merged: Dict[str, int] = {}
    for norm, rows in by_norm.items():
        if len(rows) <= 1:
            # Still normalize the lone row's stored name if needed.
            r = rows[0]
            if r.name != norm:
                print(f'  rename Word(id={r.id}, name={r.name!r}) -> {norm!r}')
                r.name = norm
            continue

        canon = _pick_canonical(rows)
        if canon.name != norm:
            print(f'  rename canonical Word(id={canon.id}, name={canon.name!r}) -> {norm!r}')
            canon.name = norm
        for r in rows:
            if r.id == canon.id:
                continue
            print(f'  MERGE Word(id={r.id}, name={r.name!r}) -> Word(id={canon.id}, name={canon.name!r})')
            # Re-target Contribution
            n_contribs = (db.query(Contribution)
                          .filter(Contribution.word_id == r.id)
                          .update({Contribution.word_id: canon.id}, synchronize_session=False))
            # Re-target DailyAssignment
            n_assign = (db.query(DailyAssignment)
                        .filter(DailyAssignment.word_id == r.id)
                        .update({DailyAssignment.word_id: canon.id}, synchronize_session=False))
            print(f'    moved {n_contribs} contributions, {n_assign} daily assignments')
            db.delete(r)
            merged[norm] = r.id
    db.commit()
    return merged


def _recount_templates(db) -> None:
    """Recompute Word.template_count from on-disk reality."""
    for w in db.query(Word).all():
        word_dir = os.path.join(TEMPLATE_DIR, w.name)
        if os.path.isdir(word_dir):
            n = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
        else:
            n = 0
        if w.template_count != n:
            print(f'  Word({w.name!r}).template_count {w.template_count} -> {n}')
            w.template_count = n
    db.commit()


def _normalize_template_dirs() -> None:
    """Rename data/templates/<NAME>/ to its normalized form when
    necessary.  On case-insensitive FS this is a metadata rename
    (`bon`/`BON` already point to the same inode).  On case-
    sensitive FS, if both forms exist we merge their files into the
    lowercase one and delete the uppercase form."""
    if not os.path.isdir(TEMPLATE_DIR):
        return
    for entry in sorted(os.listdir(TEMPLATE_DIR)):
        full = os.path.join(TEMPLATE_DIR, entry)
        if not os.path.isdir(full):
            continue
        norm = normalize_word(entry)
        if entry == norm:
            continue
        target = os.path.join(TEMPLATE_DIR, norm)
        if os.path.isdir(target) and os.path.realpath(target) != os.path.realpath(full):
            # Case-sensitive FS, two different inodes — merge files
            for fn in os.listdir(full):
                src = os.path.join(full, fn)
                dst = os.path.join(target, fn)
                if os.path.exists(dst):
                    # Both have e.g. '7.npy' — keep the larger file as
                    # tie-breaker, else skip.  Should be rare.
                    if os.path.getsize(src) > os.path.getsize(dst):
                        shutil.move(src, dst)
                    else:
                        os.remove(src)
                else:
                    shutil.move(src, dst)
            os.rmdir(full)
            print(f'  merged template dir {entry!r} into {norm!r}')
        else:
            os.rename(full, target)
            print(f'  rename template dir {entry!r} -> {norm!r}')


def main() -> int:
    db = SessionLocal()
    try:
        print('=== Step 1: rename template directories ===')
        _normalize_template_dirs()
        print('\n=== Step 2: merge duplicate Word rows ===')
        merged = _merge_words(db)
        if merged:
            print(f'  merged {len(merged)} duplicate words')
        else:
            print('  (no duplicates found)')
        print('\n=== Step 3: recompute template_count from disk ===')
        _recount_templates(db)
        print('\nMigration complete.')
    finally:
        db.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
