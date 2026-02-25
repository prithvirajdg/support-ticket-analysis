from google_play_scraper import Sort, reviews
import pandas as pd
from time import sleep
from datetime import datetime

# Porter app package IDs
APPS = {
    "porter_customer": "com.theporter.android.customerapp",
    "porter_partner": "com.theporter.android.driverapp",
}

MAX_REVIEWS = 25000      # total reviews to fetch per app
MIN_WORD_COUNT = 10      # minimum words for "meaningful"


def scrape_reviews(app_id, app_name, max_reviews=MAX_REVIEWS):
    all_reviews = []
    continuation_token = None
    batch = 0

    while len(all_reviews) < max_reviews:
        batch += 1
        print(f"[{app_name}] Fetching batch {batch}...")

        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=200,
            continuation_token=continuation_token,
        )

        if not result:
            break

        all_reviews.extend(result)
        sleep(1)

        if continuation_token is None:
            break

    df = pd.DataFrame(all_reviews)
    print(f"[{app_name}] Total fetched: {len(df)}")
    return df


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d")

    for name, pkg_id in APPS.items():
        df = scrape_reviews(pkg_id, name)

        # Add word count column
        df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))

        # Filter meaningful reviews (10+ words)
        # meaningful = df[df['word_count'] >= MIN_WORD_COUNT].copy()

        # Save all reviews to individual CSV
        filename = f"{name}_reviews_{timestamp}.csv"
        df.to_csv(filename, index=False)

        print(f"[{name}] Total reviews saved: {len(df)}")
        print(f"[{name}] Saved to: {filename}\n")
