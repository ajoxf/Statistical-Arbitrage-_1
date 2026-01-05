# Deploying the Pairs Trading Backtester

This guide shows you how to deploy the web app so others can use it.

## Option 1: Streamlit Cloud (FREE - Recommended)

The easiest way to deploy. No coding required!

### Steps:

1. **Push your code to GitHub** (if not already done)

2. **Go to Streamlit Cloud**: https://share.streamlit.io/

3. **Sign in with GitHub**

4. **Click "New app"**

5. **Fill in the form:**
   - Repository: `your-username/Statistical-Arbitrage-_1`
   - Branch: `main` (or your branch)
   - Main file path: `Statistical Arbitrage/app.py`

6. **Click "Deploy"**

7. **Share the URL** - Anyone with the link can use the app!

### Your app will be at:
`https://your-app-name.streamlit.app`

---

## Option 2: Hugging Face Spaces (FREE)

Another free option with good performance.

### Steps:

1. Go to https://huggingface.co/spaces

2. Click "Create new Space"

3. Select "Streamlit" as the SDK

4. Upload these files:
   - `app.py`
   - `requirements.txt`

5. Your app will be live!

---

## Option 3: Run Locally

For testing before deployment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at: http://localhost:8501

---

## Important Notes

- **The source code is NOT visible** to users of the deployed app
- Users can only interact with the web interface
- They cannot download or copy your Python code
- All they see is the web UI you designed

## Sharing the Link

Once deployed, simply share the URL:
- `https://your-app.streamlit.app`

Users can:
- Enter any stock tickers
- Run backtests
- View results and charts
- Download reports (if you add that feature)

Users CANNOT:
- See your source code
- Modify the strategy
- Access your GitHub repository (unless it's public)
