import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="銘柄名表示対応 ポートフォリオ最適化", layout="wide")

st.title("ポートフォリオ最適化 Webアプリ 🚀")
st.write("表のヘッダーをティッカーから会社名（正式名称）に変換して表示します。")

# --- メイン画面の設定エリア ---
st.markdown("### ⚙️ パラメーター設定")

market_type = st.radio("投資対象のメイン市場を選んでください", ["日本株 (JPYベース)", "米国株 (USDベース)"], horizontal=True)

if "favorites" not in st.session_state:
    st.session_state.favorites = ["7203.T, 6758.T, 7974.T, 8306.T", "AAPL, MSFT, NVDA, TSLA"]

col1, col2 = st.columns([3, 1])
with col2:
    selected_fav = st.selectbox("⭐ お気に入りから選ぶ", st.session_state.favorites)
with col1:
    tickers_input = st.text_input("銘柄（ティッカー）をカンマ区切りで入力", selected_fav)

if st.button("➕ 今の銘柄をお気に入りに登録"):
    if tickers_input and tickers_input not in st.session_state.favorites:
        st.session_state.favorites.append(tickers_input)
        st.success("登録しました！")

st.markdown("---")

col_left, col_right = st.columns(2)
with col_left:
    default_rf = 1.75 if "日本株" in market_type else 4.1
    risk_free_rate_pct = st.number_input(f"無リスク金利（％）", min_value=0.0, max_value=10.0, value=default_rf, step=0.01)
    rf_rate = risk_free_rate_pct / 100
with col_right:
    period_options = {"1ヶ月": "1mo", "3ヶ月": "3mo", "1年": "1y", "3年": "3y", "5年": "5y", "10年": "10y", "20年": "20y", "全期間(MAX)": "max"}
    selected_period_label = st.selectbox("データの取得期間", list(period_options.keys()), index=5)

if st.button("🚀 この条件で計算を実行する", use_container_width=True, type="primary"):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    
    with st.spinner("データを取得・分析中..."):
        try:
            # 1. 会社名（銘柄名）の取得
            ticker_names = {}
            for t in tickers:
                try:
                    info = yf.Ticker(t).info
                    # infoの中にnameがあればそれを使い、なければティッカーを表示
                    name = info.get('longName') or info.get('shortName') or t
                    ticker_names[t] = name
                except:
                    ticker_names[t] = t

            # 2. 株価データ取得
            period_val = period_options[selected_period_label]
            data = yf.download(tickers, period=period_val)["Close"]
            
            if data.empty:
                st.error("データの取得に失敗しました。")
            else:
                actual_start, actual_end = data.index[0], data.index[-1]
                st.info(f"📅 **期間:** {actual_start.strftime('%Y/%m/%d')} 〜 {actual_end.strftime('%Y/%m/%d')} ({(actual_end - actual_start).days / 365.25:.1f} 年分)")

                mu = expected_returns.mean_historical_return(data)
                S = risk_models.sample_cov(data)
                
                # 3. 最適解計算
                ef_opt = EfficientFrontier(mu, S)
                ef_opt.max_sharpe(risk_free_rate=rf_rate)
                cleaned_weights = ef_opt.clean_weights()
                opt_ret, opt_vol, opt_sharpe = ef_opt.portfolio_performance(risk_free_rate=rf_rate)
                
                # 4. ランダム点生成 (3000点に調整)
                num_portfolios = 3000
                results = np.zeros((3, num_portfolios))
                mu_array, S_matrix = mu.values, S.values
                random_hover_texts = []
                for i in range(num_portfolios):
                    w = np.random.random(len(tickers))
                    w /= np.sum(w)
                    p_ret = np.sum(w * mu_array)
                    p_vol = np.sqrt(np.dot(w.T, np.dot(S_matrix, w)))
                    results[0,i], results[1,i], results[2,i] = p_vol, p_ret, (p_ret - rf_rate) / p_vol
                    w_text = "<br>".join([f"{ticker_names.get(tickers[j], tickers[j])}: {w[j]:.1%}" for j in range(len(tickers)) if w[j] > 0.01])
                    random_hover_texts.append(f"リターン: {p_ret:.1%}<br>リスク: {p_vol:.1%}<br>比率:<br>{w_text}")
                    
                # 5. フロンティア曲線計算
                frontier_vol, frontier_ret, frontier_sharpe, frontier_weights = [], [], [], []
                risk_aversions = np.logspace(-1, 2, 80)
                for ra in risk_aversions:
                    try:
                        ef_t = EfficientFrontier(mu, S)
                        ef_t.max_quadratic_utility(risk_aversion=ra)
                        r, v, _ = ef_t.portfolio_performance(risk_free_rate=rf_rate)
                        frontier_vol.append(v); frontier_ret.append(r); frontier_sharpe.append((r - rf_rate) / v); frontier_weights.append(ef_t.clean_weights())
                    except: continue
                
                # 📊 グラフ表示
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers', marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, colorbar=dict(title='Sharpe Ratio')), name='ランダム', text=random_hover_texts, hovertemplate='%{text}<extra></extra>'))
                fig.add_trace(go.Scatter(x=frontier_vol, y=frontier_ret, mode='lines', line=dict(color='blue', width=4, shape='spline'), name='効率的フロンティア'))
                fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', marker=dict(color='red', size=20, symbol='star'), name='最適解'))
                fig.update_layout(xaxis_title="想定リスク (標準偏差)", yaxis_title="期待リターン (利回り)", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # 📋 テーブル表示（★ここを会社名に変更）
                st.markdown("---")
                st.subheader(f"📋 ポートフォリオ詳細（金利: {risk_free_rate_pct}% 設定）")
                
                f_vol, f_ret, f_sha, f_wei = list(frontier_vol) + [opt_vol], list(frontier_ret) + [opt_ret], list(frontier_sharpe) + [opt_sharpe], list(frontier_weights) + [cleaned_weights]
                combined = sorted(list(zip(f_vol, f_ret, f_sha, f_wei)), key=lambda x: x[0])
                f_vol, f_ret, f_sha, f_wei = zip(*combined)
                
                df_dict = {"想定リスク": [f"{v:.2%}" for v in f_vol], "期待リターン": [f"{r:.2%}" for r in f_ret], "シャープレシオ": [f"{s:.3f}" for s in f_sha]}
                
                # 各ティッカーのカラムを会社名で作成
                for tk in tickers:
                    display_name = ticker_names.get(tk, tk)
                    df_dict[display_name] = [f"{w.get(tk, 0):.2%}" for w in f_wei]
                
                df_frontier = pd.DataFrame(df_dict).drop_duplicates(subset=["想定リスク", "期待リターン"])
                
                opt_vol_str, opt_ret_str = f"{opt_vol:.2%}", f"{opt_ret:.2%}"
                st.dataframe(df_frontier.style.apply(lambda r: ['background-color: rgba(255, 99, 71, 0.3); font-weight: bold']*len(r) if r["想定リスク"]==opt_vol_str and r["期待リターン"]==opt_ret_str else ['']*len(r), axis=1), use_container_width=True)
                st.download_button("📥 CSVダウンロード", df_frontier.to_csv(index=False).encode('utf-8-sig'), f"frontier_data.csv")
                
        except Exception as e:
            st.error(f"エラーが発生しました。入力内容を確認してください。(詳細: {e})")
            