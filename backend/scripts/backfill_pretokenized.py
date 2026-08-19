import psycopg

from ingestion.stackoverflow_loader import CONN_STR
from retrieval.tokenize import tokenize

BATCH_SIZE = 1000


def main() -> None:
    with psycopg.connect(CONN_STR) as conn:

        with conn.cursor(name="backfill", withhold=True) as read_cur:
            read_cur.execute(
                "SELECT id, content FROM chunks WHERE content_pretokenized IS NULL"
            )
            batch = read_cur.fetchmany(BATCH_SIZE)
            total = 0

            while batch:
                with conn.cursor() as write_cur:
                    write_cur.executemany(
                        "UPDATE chunks SET content_pretokenized = %s WHERE id = %s",
                        [
                            (" ".join(tokenize(content)), chunk_id)
                            for chunk_id, content in batch
                        ],
                    )

                conn.commit()

                total += len(batch)
                print(f"backfilled {total} rows")
                batch = read_cur.fetchmany(BATCH_SIZE)


if __name__ == "__main__":
    main()
