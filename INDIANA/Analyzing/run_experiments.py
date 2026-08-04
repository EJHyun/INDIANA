
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S +0000 %Y')
    except:
        return None

print("\n[1.1] Loading NYC data and converting UTC to Local Time (UTC-5)...")
nyc_df = pd.read_csv('TSMC2014_NYC_UTF-8.txt', sep='\t', header=None,
                     names=['UserID', 'VenueID', 'VenueCategoryID', 'VenueCategory',
                            'Latitude', 'Longitude', 'TimezoneOffset', 'UTCTimestamp'])
nyc_df['UTCTimestamp'] = nyc_df['UTCTimestamp'].apply(parse_timestamp)
nyc_df = nyc_df.dropna(subset=['UTCTimestamp'])
nyc_df['LocalTimestamp'] = nyc_df['UTCTimestamp'] - timedelta(hours=5)
nyc_df['UserID'] = 'NYC_' + nyc_df['UserID'].astype(str)
print(f"   Loaded {len(nyc_df):,} NYC records")

print("\n[1.2] Loading TKY data and converting UTC to Local Time (UTC+9)...")
tky_df = pd.read_csv('TSMC2014_TKY_UTF-8.txt', sep='\t', header=None,
                     names=['UserID', 'VenueID', 'VenueCategoryID', 'VenueCategory',
                            'Latitude', 'Longitude', 'TimezoneOffset', 'UTCTimestamp'])
tky_df['UTCTimestamp'] = tky_df['UTCTimestamp'].apply(parse_timestamp)
tky_df = tky_df.dropna(subset=['UTCTimestamp'])
tky_df['LocalTimestamp'] = tky_df['UTCTimestamp'] + timedelta(hours=9)
tky_df['UserID'] = 'TKY_' + tky_df['UserID'].astype(str)
print(f"   Loaded {len(tky_df):,} TKY records")

print("\n[1.3] Concatenating datasets and extracting local hour...")
df = pd.concat([nyc_df, tky_df], ignore_index=True)
df['hour'] = df['LocalTimestamp'].dt.hour
print(f"   Total records: {len(df):,}")

print("\n[1.4] Generating next_category transitions (shift -1 by UserID)...")
df = df.sort_values(['UserID', 'LocalTimestamp']).reset_index(drop=True)
df['next_category'] = df.groupby('UserID')['VenueCategory'].shift(-1)

stop_list = ['Road', 'Building', 'Bridge', 'Others', 'Home (private)', 'City', 'Moving Target']
print(f"\n[1.5] Filtering stop-list categories: {stop_list}")
before_filter = len(df)
df = df[
    (~df['VenueCategory'].isin(stop_list)) &
    (~df['next_category'].isin(stop_list)) &
    (df['next_category'].notna())
].copy()
print(f"   Records before filter: {before_filter:,}")
print(f"   Records after filter: {len(df):,}")
print(f"   Removed: {before_filter - len(df):,} records")

df['source'] = df['VenueCategory']
df['target'] = df['next_category']

print("\n[1.6] Data preprocessing complete!")
print(f"   Final dataset size: {len(df):,} transitions")
print(f"   Unique users: {df['UserID'].nunique():,}")
print(f"   Unique source categories: {df['source'].nunique():,}")


RUN_EXP1 = False
RUN_EXP2 = False
RUN_EXP3 = False
RUN_EXP4 = True

Bagel_Targets = ['High School', 'Gym / Fitness Center']
Subway_Targets = ['University']
Exp1_Sources = ['Animal Shelter', 'Arts & Entertainment', 'Airport', 'Department Store']
Exp1_Targets = ['Salad Place', 'Comedy Club', 'Coffee Shop', 'French Restaurant', 'Hotel', 'Cosmetics Shop', 'Gym / Fitness Center', 'High School']
Bagel_Targets = Exp1_Targets
Subway_Targets = Exp1_Targets


Exp2_Sources = [
    'Animal Shelter',
    'Arts & Entertainment',
    'Airport',
    'Department Store',
]


Exp2_DefaultTargets = [
    'Salad Place',
    'Comedy Club',
    'Coffee Shop',
    'French Restaurant',
    'Hotel',
    'Cosmetics Shop',
    'Gym / Fitness Center',
    'High School',
]


Exp2_Alternatives = {
    'Subway': ['Train Station'], 
}

print(f"\nBagel_Targets: {Bagel_Targets}")
print(f"Subway_Targets: {Subway_Targets}")
print(f"Exp1_Sources: {Exp1_Sources}")
print(f"Exp1_Targets: {Exp1_Targets}")
print(f"\nExp2_Sources : {Exp2_Sources}")
print(f"Exp2_DefaultTargets : {Exp2_DefaultTargets}")


def calculate_lift(df_transitions, source, target):
    s_to_t = len(df_transitions[(df_transitions['source'] == source) & 
                                 (df_transitions['target'] == target)])
    
    s_total = len(df_transitions[df_transitions['source'] == source])
    
    if s_total == 0:
        return np.nan
    
    p_t_given_s = s_to_t / s_total
    
    t_total = len(df_transitions[df_transitions['target'] == target])
    total_transitions = len(df_transitions)
    
    if total_transitions == 0:
        return np.nan
    
    p_t = t_total / total_transitions
    
    if p_t == 0:
        return np.nan
    
    lift = p_t_given_s / p_t
    return lift

lift_df = None

if RUN_EXP1:
    print("\n[3.1] Calculating lift matrix...")
    lift_matrix = np.zeros((len(Exp1_Targets), len(Exp1_Sources)))

    for i, target in enumerate(Exp1_Targets):
        for j, source in enumerate(Exp1_Sources):
            lift_val = calculate_lift(df, source, target)
            lift_matrix[i, j] = lift_val

    lift_df = pd.DataFrame(lift_matrix, index=Exp1_Targets, columns=Exp1_Sources)

    print("\n[3.2] Creating heatmap visualization...")
    final_folder = 'final'
    os.makedirs(final_folder, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(lift_df, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=5,
                cbar_kws={'label': 'Lift Value'}, ax=ax, linewidths=0.5)
    ax.set_title('Experiment 1: Static Lift Analysis\n(Lift = P(Target|Source) / P(Target))', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Source Category', fontsize=12)
    ax.set_ylabel('Target Category', fontsize=12)
    plt.tight_layout()
    filename = os.path.join(final_folder, 'result_exp1_lift.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()
else:
    print("\n[3.X] Skipping Experiment 1 (RUN_EXP1=False)")


if RUN_EXP2:
    print("\n" + "=" * 80)
    print("PART 4: Experiment 2 - Temporal Divergence")
    print("=" * 80)

    if lift_df is None:
        print("\n[4.0] Exp1 was not run. Calculating lift matrix for exp2...")
        lift_matrix = np.zeros((len(Exp1_Targets), len(Exp1_Sources)))
        
        for i, target in enumerate(Exp1_Targets):
            for j, source in enumerate(Exp1_Sources):
                lift_val = calculate_lift(df, source, target)
                lift_matrix[i, j] = lift_val
        
        lift_df = pd.DataFrame(lift_matrix, index=Exp1_Targets, columns=Exp1_Sources)
        print("   Lift matrix calculated for exp2 use.")

    final_folder = 'final'
    os.makedirs(final_folder, exist_ok=True)
    print(f"\n[4.1] Created output folder: {final_folder}/")

    def analyze_temporal_divergence(df_transitions, source_name, targets_list, alternatives=None, output_folder='final'):
        source_data = df_transitions[df_transitions['source'] == source_name].copy()
        actual_source_name = source_name
        
        if len(source_data) < 100 and alternatives is not None:
            for alt in alternatives:
                alt_data = df_transitions[df_transitions['source'] == alt].copy()
                if len(alt_data) > len(source_data):
                    source_data = alt_data
                    actual_source_name = alt
                    print(f"   Using '{alt}' instead of '{source_name}' (found {len(alt_data):,} transitions)")
        
        print(f"   {actual_source_name} transitions: {len(source_data):,}")
        
        if len(source_data) > 0:
            lift_scores = {}
            for target in targets_list:
                if target != actual_source_name:
                    lift_val = calculate_lift(df_transitions, actual_source_name, target)
                    if not np.isnan(lift_val):
                        lift_scores[target] = lift_val
            
            top_10_targets = sorted(lift_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            top_10_targets = [target for target, _ in top_10_targets]
            
            if len(top_10_targets) < 10:
                top_10_targets = [target for target in targets_list if target in lift_scores and target != actual_source_name]
            
            print(f"   Top 10 targets by lift: {top_10_targets[:5]}... (showing {len(top_10_targets)} targets)")
            
            probs_all = np.zeros((len(targets_list), 24))
            
            for h in range(24):
                hour_data = source_data[source_data['hour'] == h]
                total_at_hour = len(hour_data)
                
                if total_at_hour > 0:
                    for i, target in enumerate(targets_list):
                        count = len(hour_data[hour_data['target'] == target])
                        probs_all[i, h] = count / total_at_hour
            
            top_10_indices = [targets_list.index(target) for target in top_10_targets if target in targets_list]
            probs = probs_all[top_10_indices, :]
            
            prob_df = pd.DataFrame(probs, index=top_10_targets, columns=range(24))
            
            fig, ax = plt.subplots(figsize=(14, 8))
            max_prob_value = 0.0
            sns.heatmap(prob_df, annot=False, cmap='YlOrRd', vmin=0, vmax=1,
                        cbar_kws={'label': 'Lift value'}, ax=ax, linewidths=0.5)
            ax.set_title(f'Experiment 2: {actual_source_name} → Target Transitions by Local Hour\n(P(Target | Hour)) - Top 10 by Lift', 
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Local Hour (0-23)', fontsize=12)
            ax.set_ylabel('Target Category (Top 10 by Lift)', fontsize=12)
            
            safe_name = actual_source_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(output_folder, f'result_exp2_{safe_name.lower()}.png')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   Saved: {filename}")
            plt.close()
            return filename
        else:
            print(f"   WARNING: No {actual_source_name} transitions found! Creating empty visualization...")
            lift_scores = {}
            for target in targets_list:
                if target != actual_source_name:
                    lift_val = calculate_lift(df_transitions, actual_source_name, target)
                    if not np.isnan(lift_val):
                        lift_scores[target] = lift_val
            
            top_10_targets = sorted(lift_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            top_10_targets = [target for target, _ in top_10_targets]
            
            if len(top_10_targets) < 10:
                top_10_targets = [target for target in targets_list if target != actual_source_name][:10]
            
            prob_df = pd.DataFrame(np.zeros((len(top_10_targets), 24)), 
                                  index=top_10_targets, columns=range(24))
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(prob_df, annot=False, cmap='YlOrRd', vmin=0, vmax=1,
                        cbar_kws={'label': 'Transition Probability'}, ax=ax, linewidths=0.5)
            ax.set_title(f'Experiment 2: {actual_source_name} → Target Transitions by Local Hour\n(P(Target | Hour)) - NO DATA - Top 10 by Lift', 
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Local Hour (0-23)', fontsize=12)
            ax.set_ylabel('Target Category (Top 10 by Lift)', fontsize=12)
            
            safe_name = actual_source_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(output_folder, f'result_exp2_{safe_name.lower()}.png')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   Saved: {filename} (empty - no data)")
            plt.close()
            return filename

    generated_files = []
    for idx, source in enumerate(Exp2_Sources, 1):
        print(f"\n[4.{idx+1}] Analyzing: {source}...")
        
        if source in lift_df.columns:
            source_lifts = lift_df[source]
            valid_targets = source_lifts[source_lifts > 0].index.tolist()
            targets = valid_targets
            print(f"   Found {len(targets)} targets with lift > 0 from exp1 results")
        else:
            targets = Exp2_Targets.get(source, Exp2_DefaultTargets)
            print(f"   Source not in exp1, using {'custom targets' if source in Exp2_Targets else 'default targets'}")
        
        alternatives = Exp2_Alternatives.get(source, None)
        
        filename = analyze_temporal_divergence(df, source, targets, alternatives, final_folder)
        generated_files.append(filename)

    print(f"\n[4.X] Experiment 2 complete! Generated {len(generated_files)} visualization(s).")
else:
    print("\n[4.X] Skipping Experiment 2 (RUN_EXP2=False)")


if RUN_EXP4:
    print("\n" + "=" * 80)
    print("PART 5: Experiment 4 - Temporal Conditional Probability Line Chart")
    print("=" * 80)

    if lift_df is None:
        print("\n[5.0] Exp1 was not run. Calculating lift matrix for exp4...")
        lift_matrix = np.zeros((len(Exp1_Targets), len(Exp1_Sources)))
        
        for i, target in enumerate(Exp1_Targets):
            for j, source in enumerate(Exp1_Sources):
                lift_val = calculate_lift(df, source, target)
                lift_matrix[i, j] = lift_val
        
        lift_df = pd.DataFrame(lift_matrix, index=Exp1_Targets, columns=Exp1_Sources)
        print("   Lift matrix calculated for exp4 use.")

    final_folder = 'final'
    os.makedirs(final_folder, exist_ok=True)
    print(f"\n[5.1] Created output folder: {final_folder}/")

    def calculate_conditional_probability_by_hour(df_transitions, source, target, hour):
        at_hour = df_transitions[df_transitions['hour'] == hour]
        total_at_hour = len(at_hour)

        if total_at_hour == 0:
            return np.nan

        source_at_hour = at_hour[at_hour['source'] == source]
        s_total_at_hour = len(source_at_hour)

        if s_total_at_hour == 0:
            return np.nan

        p_t_given_s_h = len(source_at_hour[source_at_hour['target'] == target]) / s_total_at_hour

        p_t_given_h = len(at_hour[at_hour['target'] == target]) / total_at_hour

        if p_t_given_h == 0:
            return np.nan

        return p_t_given_s_h / p_t_given_h

    def analyze_temporal_conditional_prob_linechart(df_transitions, source_name, targets_list, alternatives=None, output_folder='final', target_color_map=None):
        source_data = df_transitions[df_transitions['source'] == source_name].copy()
        actual_source_name = source_name
        
        if len(source_data) < 100 and alternatives is not None:
            for alt in alternatives:
                alt_data = df_transitions[df_transitions['source'] == alt].copy()
                if len(alt_data) > len(source_data):
                    source_data = alt_data
                    actual_source_name = alt
                    print(f"   Using '{alt}' instead of '{source_name}' (found {len(alt_data):,} transitions)")
        
        print(f"   {actual_source_name} transitions: {len(source_data):,}")
        
        if len(source_data) > 0:
            valid_targets = [target for target in targets_list if target != actual_source_name]
            
            print(f"   Using {len(valid_targets)} targets from Exp1_Targets (excluding source itself)")
            
            hours = list(range(24))
            prob_data = {}
            
            for target in valid_targets:
                prob_by_hour = []
                for h in hours:
                    prob_val = calculate_conditional_probability_by_hour(df_transitions, actual_source_name, target, h)
                    prob_by_hour.append(prob_val)
                prob_data[target] = prob_by_hour
            
            fig, ax = plt.subplots(figsize=(14, 8))
            max_prob_value = 0.0
            
            for target in valid_targets:
                prob_values = prob_data[target]
                prob_values_clean = [0 if np.isnan(v) else v for v in prob_values]
                
                prob_series = pd.Series(prob_values_clean, index=hours)
                prob_smoothed = prob_series.rolling(window=3, center=True, min_periods=1).mean()
                
                if target_color_map and target in target_color_map:
                    target_color = target_color_map[target]
                else:
                    target_color = 'gray'
                
                ax.plot(hours, prob_smoothed.values, marker='o', label=target, 
                       color=target_color, linewidth=2, markersize=4)
                max_prob_value = max(max_prob_value, prob_smoothed.max())
            
            ax.set_xlabel('(Hour)', fontsize=12)
            ax.set_ylabel('Lift value', fontsize=12)
            ax.set_title(f'Experiment 4: {actual_source_name} → Target Conditional Probability by Hour\n(P(Next=Target | Current={actual_source_name}, Hour)) - Exp1 Targets', 
                         fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, 23.5)
            y_upper = (max_prob_value if not np.isnan(max_prob_value) else 0) + 1.0
            ax.set_ylim(0, y_upper if y_upper > 0 else 0.1)
            ax.set_xticks(hours)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            
            safe_name = actual_source_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(output_folder, f'result_exp4_{safe_name.lower()}.png')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   Saved: {filename}")
            plt.close()
            return filename
        else:
            print(f"   WARNING: No {actual_source_name} transitions found! Creating empty visualization...")
            valid_targets = [target for target in targets_list if target != actual_source_name]
            
            hours = list(range(24))
            fig, ax = plt.subplots(figsize=(14, 8))
            max_prob_value = 0.0
            
            for target in valid_targets:
                if target_color_map and target in target_color_map:
                    target_color = target_color_map[target]
                else:
                    target_color = 'gray'
                
                ax.plot(hours, [0] * 24, marker='o', label=target, 
                       color=target_color, linewidth=2, markersize=4)
            
            ax.set_xlabel('(Hour)', fontsize=12)
            ax.set_ylabel('Lift value', fontsize=12)
            ax.set_title(f'Experiment 4: {actual_source_name} → Target Conditional Probability by Hour\n(P(Next=Target | Current={actual_source_name}, Hour)) - NO DATA - Exp1 Targets', 
                         fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, 23.5)
            y_upper = (max_prob_value if not np.isnan(max_prob_value) else 0) + 1.0
            ax.set_ylim(0, y_upper if y_upper > 0 else 0.1)
            ax.set_xticks(hours)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            
            safe_name = actual_source_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = os.path.join(output_folder, f'result_exp4_{safe_name.lower()}.png')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   Saved: {filename} (empty - no data)")
            plt.close()
            return filename

    print(f"\n[5.2] Creating consistent color map for Exp1_Targets...")
    exp1_targets_colors = plt.cm.tab10(np.linspace(0, 1, len(Exp1_Targets)))
    target_color_map = {target: exp1_targets_colors[i] for i, target in enumerate(Exp1_Targets)}
    print(f"   Color map created for {len(Exp1_Targets)} targets")
    
    generated_files = []
    for idx, source in enumerate(Exp2_Sources, 1):
        print(f"\n[5.{idx+2}] Analyzing: {source}...")
        
        targets = Exp1_Targets
        print(f"   Using Exp1_Targets: {len(targets)} targets")
        
        alternatives = Exp2_Alternatives.get(source, None)
        
        filename = analyze_temporal_conditional_prob_linechart(df, source, targets, alternatives, final_folder, target_color_map)
        generated_files.append(filename)

    print(f"\n[5.X] Experiment 4 complete! Generated {len(generated_files)} visualization(s).")
else:
    print("\n[5.X] Skipping Experiment 4 (RUN_EXP4=False)")

print("\n" + "=" * 80)
print("PART 6: Experiment 3 - Per-Category Lift (Top 10 Targets)")
print("=" * 80)

def generate_exp3_lift_heatmap(df_transitions, source_name, all_targets, output_folder='exp3'):
    lift_scores = {}
    for tgt in all_targets:
        if tgt == source_name:
            continue
        lift_val = calculate_lift(df_transitions, source_name, tgt)
        if not np.isnan(lift_val):
            lift_scores[tgt] = lift_val

    top_targets = sorted(lift_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_targets = [t for t, _ in top_targets]

    if len(top_targets) == 0:
        print(f"   WARNING: No targets found for {source_name} (after exclusion). Skipping plot.")
        return None

    data = []
    for tgt in top_targets:
        data.append([lift_scores.get(tgt, 0.0)])

    heatmap_df = pd.DataFrame(data, index=top_targets, columns=[source_name])

    fig, ax = plt.subplots(figsize=(4, max(6, len(top_targets) * 0.35)))
    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0,
                cbar_kws={'label': 'Lift P(T|S)/P(T)'}, ax=ax, linewidths=0.5)
    ax.set_title(f'Experiment 3: {source_name}\nTop 10 Targets by Lift', fontsize=12, fontweight='bold')
    ax.set_xlabel('Source', fontsize=10)
    ax.set_ylabel('Target (Top 10 by Lift)', fontsize=10)

    safe_name = source_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    filename = os.path.join(output_folder, f'result_exp3_{safe_name.lower()}.png')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()
    return filename

if RUN_EXP3:
    exp3_folder = 'exp3'
    os.makedirs(exp3_folder, exist_ok=True)
    print(f"\n[5.0] Created output folder: {exp3_folder}/")

    exp3_sources = sorted(df['source'].unique().tolist())
    exp3_targets = sorted(df['target'].unique().tolist())

    exp3_files = []
    for idx, src in enumerate(exp3_sources, 1):
        print(f"\n[5.{idx}] Computing lift top-10 targets for: {src}")
        fname = generate_exp3_lift_heatmap(df, src, exp3_targets, exp3_folder)
        if fname:
            exp3_files.append(fname)

    print(f"\n[5.X] Experiment 3 complete! Generated {len(exp3_files)} visualization(s).")
else:
    print("\n[5.X] Skipping Experiment 3 (RUN_EXP3=False)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
