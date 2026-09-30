# BigQuery -> CSV

import argparse
import csv
import os
import sys

from google.cloud import bigquery

TABLE = "patents-public-data.patents.publications"

QUERY = """
WITH kr AS (
  SELECT
    family_id,
    publication_number AS kr_pub,
    (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'ko' LIMIT 1) AS ko_abstract
  FROM `{table}`
  WHERE country_code = 'KR'
    AND CAST(family_id AS STRING) NOT IN ('-1', '0')
    AND EXISTS (SELECT 1 FROM UNNEST(abstract_localized) WHERE language = 'ko')
),
foreign AS (
  SELECT
    family_id,
    publication_number AS en_pub,
    country_code,
    (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS en_abstract
  FROM `{table}`
  WHERE country_code IN ('US', 'WO')
    AND CAST(family_id AS STRING) NOT IN ('-1', '0')
    AND EXISTS (SELECT 1 FROM UNNEST(abstract_localized) WHERE language = 'en')
),
joined AS (
  SELECT
    family_id,
    kr.kr_pub, foreign.en_pub, foreign.country_code AS en_source,
    kr.ko_abstract, foreign.en_abstract,
    -- family_id 기준: 같은 출원의 공개본/등록본이 각각 통과하던 문제를 여기서 막는다.
    -- kr_pub, en_pub 정렬은 동점일 때 재실행해도 같은 행이 뽑히게 하기 위한 것.
    ROW_NUMBER() OVER (
      PARTITION BY family_id
      ORDER BY IF(foreign.country_code = 'WO', 0, 1), kr.kr_pub, foreign.en_pub
    ) AS rn
  FROM kr
  JOIN foreign USING (family_id)
  WHERE kr.ko_abstract IS NOT NULL
    AND foreign.en_abstract IS NOT NULL
    AND LENGTH(kr.ko_abstract) > 20
    AND LENGTH(foreign.en_abstract) > 20
)
SELECT family_id, kr_pub, en_pub, en_source, ko_abstract, en_abstract
FROM joined
WHERE rn = 1
-- Hash order instead of RAND(): a well-mixed sample that is identical on every rerun.
-- (The published splits predate this change, so they came from a RAND() sample; the
-- HF dataset, not a rerun of this query, is the canonical copy of them.)
ORDER BY FARM_FINGERPRINT(CAST(family_id AS STRING))
LIMIT @sample_limit
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    ap.add_argument("--sample-limit", type=int, default=30000)
    ap.add_argument("--out", default="data/raw/patent_pairs.csv")  # where prep_dataset.py reads
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.project:
        sys.exit("No GCP project: pass --project or set GOOGLE_CLOUD_PROJECT")

    client = bigquery.Client(project=args.project)
    query = QUERY.format(table=TABLE)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("sample_limit", "INT64", args.sample_limit)],
        dry_run=args.dry_run,
        use_query_cache=not args.dry_run,
    )

    job = client.query(query, job_config=job_config)

    if args.dry_run:
        gb = job.total_bytes_processed / 1e9
        print(f"Estimated bytes scanned: {gb:.2f} GB "
              f"({gb / 1000:.4f} TB of your 1 TB free quota)")
        return

    rows = list(job.result())
    print(f"Fetched {len(rows)} KR<->EN pairs. "
          f"Bytes billed: {job.total_bytes_billed / 1e9:.2f} GB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family_id", "kr_pub", "en_pub", "en_source", "ko_abstract", "en_abstract"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "family_id": r["family_id"],
                "kr_pub": r["kr_pub"],
                "en_pub": r["en_pub"],
                "en_source": r["en_source"],
                "ko_abstract": r["ko_abstract"],
                "en_abstract": r["en_abstract"],
            })

    print(f"Wrote {args.out} -- next: python data_pipeline/prep_dataset.py")

if __name__ == "__main__":
    main()