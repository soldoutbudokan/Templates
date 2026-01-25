# DealBot

Automated deal finder that searches the web daily and sends push notifications when prices drop below your target.

**Features:**
- Searches the entire web via Brave Search API
- Direct price scraping for Shopify stores
- Push notifications via Ntfy (free, no account needed)
- Runs daily on GitHub Actions (free)
- Blocklist for second-hand/sketchy sites

## Quick Setup (10 minutes)

### 1. Get API Key

1. Go to [api.search.brave.com](https://api.search.brave.com)
2. Create free account → Get API key
3. Free tier: 2,000 searches/month

### 2. Setup Notifications

1. Install **Ntfy** app: [Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) | [iOS](https://apps.apple.com/app/ntfy/id1625396347)
2. Open app → Subscribe to a unique topic (e.g., `dealbot-yourname-123`)

### 3. Configure Products

Edit `dealbot.py` and update the `PRODUCTS` list:

```python
PRODUCTS = [
    {
        "name": "Product Display Name",
        "keywords": "search keywords for brave",
        "url": "",  # Optional: direct URL for Shopify stores
        "typical_price": 100.00,
        "target_price": 80.00,
    },
]
```

### 4. Deploy to GitHub

1. Create a new GitHub repo
2. Push this folder to the repo
3. Go to **Settings → Secrets → Actions**
4. Add two secrets:
   - `BRAVE_API_KEY`: Your Brave API key
   - `NTFY_TOPIC`: Your Ntfy topic name

### 5. Test

1. Go to **Actions** tab
2. Click **DealBot Daily Check**
3. Click **Run workflow**
4. Check your phone for notification!

## Configuration

### Products

| Field | Description |
|-------|-------------|
| `name` | Display name in notifications |
| `keywords` | Search terms (be specific) |
| `url` | Optional direct URL (works great for Shopify stores) |
| `typical_price` | Normal retail price |
| `target_price` | Notify when price drops below this |

### Blocked Sites

Edit `BLOCKED_SITES` to ignore certain domains:

```python
BLOCKED_SITES = [
    "ebay.ca",
    "ebay.com",
    "kijiji.ca",
    # Add more...
]
```

### Schedule

Default: Runs daily at 2am EST. Edit `.github/workflows/dealbot.yml` to change:

```yaml
schedule:
  - cron: '0 7 * * *'  # 7am UTC = 2am EST
```

## Files

```
dealbot/
├── dealbot.py                    # Main script - edit products here
├── .github/workflows/dealbot.yml # GitHub Actions schedule
├── .gitignore
└── README.md
```

## Cost

**$0/month** - Everything runs on free tiers:
- GitHub Actions: Free for public/private repos
- Brave Search: 2,000 queries/month free
- Ntfy: Unlimited, free forever
