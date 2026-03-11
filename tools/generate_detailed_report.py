import os
import sys
import json
import pandas as pd
import logging

# Add root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(root_dir, "weights")
CONFIGS_DIR = os.path.join(root_dir, "configs")
METRICS_CSV = os.path.join(root_dir, "model_metrics_summary.csv")

def find_config_for_model(model_name, model_dir):
    """
    Heuristic to find the correct config file for a model.
    """
    # Handle rank_candidate prefix
    if model_name.startswith("rank_candidate_"):
        real_name = model_name.replace("rank_candidate_", "")
        # Try to find in rank_candidates dir
        # We don't know the exact subplot (archived/etc), so we walk
        rank_dir = os.path.join(MODELS_DIR, "rank_candidates")
        for root, dirs, files in os.walk(rank_dir):
            if os.path.basename(root) == real_name:
                local_config = os.path.join(root, "config.json")
                if os.path.exists(local_config):
                    return local_config
    
    # 1. Config in model dir
    local_config = os.path.join(model_dir, "config.json")
    if os.path.exists(local_config):
        return local_config
    
    # 2. Config in zoo
    zoo_config = os.path.join(CONFIGS_DIR, "zoo", f"{model_name}.json")
    if os.path.exists(zoo_config):
        return zoo_config

    # 3. Recursive search in configs dir
    for root, dirs, files in os.walk(CONFIGS_DIR):
        if f"{model_name}.json" in files:
            return os.path.join(root, f"{model_name}.json")
            
    return None

def analyze_model(row, config):
    """
    Generate analysis text based on metrics and config.
    """
    h1_acc = row.get('H1_Accuracy', 0)
    h2_return = row.get('H2_Return', 0)
    market_return = row.get('H2_Market_Return', 0)
    
    alpha = h2_return - market_return
    
    classification = "Unknown"
    if alpha > 0.05:
        classification = "**Alpha Generator** (Beats Market)"
    elif abs(alpha) < 0.05:
        classification = "**Market Tracker** (Matches Market)"
    else:
        classification = "**Underperformer** (Lags Market)"
        
    strengths = []
    weaknesses = []
    
    # Analyze Accuracy vs Profit
    if h1_acc > 0.58 and h2_return < market_return:
        weaknesses.append("High Accuracy Trap: Predicts 'Neutral' too often, missing trends.")
    
    if h2_return > market_return:
        strengths.append(f"Strong Trend Capturing (Alpha: {alpha*100:.1f}%)")
    
    bear_recall = row.get('H1_Recall_Bear', 0)
    if bear_recall > 0.3:
        strengths.append("High Bear Sensitivity (Good for hedging)")
    elif bear_recall < 0.1:
        weaknesses.append("Blind to Bear Markets (High downside risk)")

    return classification, strengths, weaknesses

def main():
    if not os.path.exists(METRICS_CSV):
        logger.error(f"Metrics CSV not found: {METRICS_CSV}")
        return

    df = pd.read_csv(METRICS_CSV)
    
    # Calculate Alpha for sorting
    df['Alpha'] = df['H2_Return'] - df['H2_Market_Return']
    df.sort_values(by='Alpha', ascending=False, inplace=True)
    
    report = []
    report.append("# Detailed Model Catalogue")
    report.append("A comprehensive technical and performance analysis of all available models.\n")
    
    report.append("## Executive Summary")
    report.append("| Model | Persona | H2 Return | Alpha | Pro Grade? |")
    report.append("| :--- | :--- | :--- | :--- | :--- |")
    
    model_cards = []
    
    for _, row in df.iterrows():
        model_name = row['Model']
        model_dir = os.path.join(MODELS_DIR, model_name)
        
        # Load Config
        config_path = find_config_for_model(model_name, model_dir)
        if not config_path:
            logger.warning(f"Config not found for {model_name}")
            config = {}
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
        # Analysis
        persona, strengths, weaknesses = analyze_model(row, config)
        
        h2_return = row.get('H2_Return', 0)
        market_return = row.get('H2_Market_Return', 0)
        alpha = h2_return - market_return
        pro_grade = "✅ YES" if alpha > 0 else "❌ NO"
        
        report.append(f"| {model_name} | {persona.split('**')[1]} | {h2_return*100:.1f}% | {alpha*100:.1f}% | {pro_grade} |")
        
        # Build Model Card
        card = []
        card.append(f"## 📦 {model_name}")
        card.append(f"**Persona**: {persona}")
        card.append("")
        
        card.append("### 📊 Performance Summary")
        card.append(f"- **H1 (Short Term)**: Acc: {row['H1_Accuracy']*100:.1f}% | Return: {row['H1_Return']*100:.1f}%")
        card.append(f"- **H2 (Medium Term)**: Acc: {row['H2_Accuracy']*100:.1f}% | Return: {row['H2_Return']*100:.1f}% (Market: {market_return*100:.1f}%)")
        card.append(f"- **Bear Recall (Risk)**: {row['H1_Recall_Bear']*100:.1f}%")
        
        card.append("")
        card.append("### 💪 Strengths & Weaknesses")
        if strengths:
            for s in strengths: card.append(f"- ✅ {s}")
        else:
            card.append("- (None Identified)")
            
        if weaknesses:
            for w in weaknesses: card.append(f"- ⚠️ {w}")
        else:
            card.append("- (None Identified)")
            
        card.append("")
        card.append("### ⚙️ Technical Specifications")
        
        # Architecture
        arch = config.get('model_type', 'Unknown')
        layers = config.get('lstm_layers', '?')
        card.append(f"- **Architecture**: {arch} ({layers} LSTM Layers)")
        card.append(f"- **Lookback Window**: {config.get('seq_length', '?')} days")
        
        # Inputs
        features = config.get('feature_pipeline', [])
        if not features:
            features = config.get('feature_cols', [])
            
        card.append(f"- **Input Features ({len(features)})**:")
        
        # List a few key features carefully
        feature_list = []
        if isinstance(features, list) and len(features) > 0:
            if isinstance(features[0], dict):
                # Pipeline format
                for f in features:
                    if 'input' in f: feature_list.append(f.get('input'))
                    elif 'indicator' in f: feature_list.append(f.get('indicator'))
            else:
                # Simple list format
                feature_list = features
        
        # Deduplicate and limit
        unique_features = sorted(list(set([str(f) for f in feature_list])))
        card.append(f"  - {', '.join(unique_features[:10])} ...")

        # Outputs
        targets = config.get('targets', [])
        card.append(f"- **Outputs**:")
        if targets:
            for t in targets:
                card.append(f"  - **{t.get('name')}**: {t.get('type')} ({t.get('horizon')} days)")
        else:
            card.append("  - (Standard H1/H2 Regression Targets)")
            
        card.append("---")
        model_cards.append("\n".join(card))

    report.append("\n")
    report.extend(model_cards)
    
    with open("model_catalogue.md", "w", encoding='utf-8') as f:
        f.write("\n".join(report))
    logger.info("Detailed catalogue saved to model_catalogue.md")

if __name__ == "__main__":
    main()
