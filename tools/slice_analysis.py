import os
import shura
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from model.svm_classifier import SVMClassifier
from data_loader.utils import map_brain_areas, load_config
from data_loader.pre_process import build_data_dict


# --- Setup Logger ---
log = shura.get_logger(name= "Slice Analysis (Tool)", level="DEBUG", to_file= False, filename="slice_analysis.log", file_format="log")



def full_slice_analysis(directory, output_dir,config, slices=None):
    brain_map = map_brain_areas(directory)

    if slices is None:
        slices = ['start', 'middle', 'end']

    df_results = []

    for hemisphere in brain_map:
        for net in brain_map[hemisphere]:
            for sub_area in brain_map[hemisphere][net]:
                for index in brain_map[hemisphere][net][sub_area]:
                    for slice in slices:
                        try:
                            log.info(f"Processing: {hemisphere}_{net}_{sub_area}_{index}_{slice}")

                            svm_dict = build_data_dict(
                                directory=config["directory"],
                                NET=net,
                                SUB_AREA=sub_area,
                                idx=index,
                                H=hemisphere,
                                slice=slice,
                                dur=config["dur"],
                                offset=config["offset"],
                                z_norm=config["z_norm"],
                                is_rest=config["is_rest"]
                            )

                            if "data" in svm_dict:
                                X, y = svm_dict["data"]
                                clf = SVMClassifier(
                                    kernel=config["kernel"],
                                    scale=config["scale"],
                                    k_folds=config["k_folds"]
                                )
                                accs, reports, best_mat, sum_mat =  clf.train_and_evaluate(X, y)
                            else:
                                log.warning("Failed to load data from pickle file.")


                            df_results.append({
                                'Hemisphere': hemisphere,
                                'Network': net,
                                'Sub_Area': sub_area,
                                'Index': index,
                                'Slice': slice,
                                'Mean_Accuracy': np.mean(accs),
                                'Std_Accuracy': np.std(accs),
                                'Max_Accuracy': np.max(accs),
                                'Min_Accuracy': np.min(accs),
                                'SEM_TEST': np.std(accs)/np.sqrt(config["k_folds"]),
                                'Accuracies': accs
                            })

                        except Exception as e:
                            log.error(f"Error on {hemisphere}_{net}_{sub_area}_{index}_{slice}")
                            raise e

    df = pd.DataFrame(df_results)
    df = df.sort_values(by=['Mean_Accuracy'], ascending=False)

    output_path = os.path.join(output_dir, f"slice_analysis_results_dur{config["dur"]}_offset{config["offset"]}_{config["k_folds"]}fold_rest{config["is_rest"]}.csv")
    df.to_csv(output_path, index=False)
    log.info(f"Saved results to {output_path}")

    return df

def plot_slice_analysis(config, output_dir, df):
    df['SEM'] = df['SEM_TEST']
    # Unique networks in the data
    networks = df['Network'].unique()

    # Loop through each network and create a plot
    for network in networks:
        network_data = df[df['Network'] == network]

        # Check if Sub_Area column has non-empty values for the network
        if network_data['Sub_Area'].notna().any():
            group_by_columns = ['Hemisphere', 'Sub_Area', 'Index']
            x_label_format = lambda row: f"{row['Hemisphere']}_{row['Sub_Area']}_{row['Index']}"
            x_label = 'Brain Sub-Areas (Hemisphere_SubArea_Index)'
        else:
            group_by_columns = ['Hemisphere', 'Index']
            x_label_format = lambda row: f"{row['Hemisphere']}_{row['Index']}"
            x_label = 'Brain Sub-Areas (Hemisphere_Index)'

        # Pivot data
        plot_data = network_data.pivot_table(
            index=group_by_columns,
            columns='Slice',
            values=['Mean_Accuracy', 'SEM']
        ).reset_index()
        plot_data.columns = ['_'.join(filter(None, col)).strip() for col in plot_data.columns]
        # Calculate percentage growth
        plot_data['Growth_Percentage'] = 100 * (plot_data['Mean_Accuracy_end'] - plot_data['Mean_Accuracy_start']) / \
                                         plot_data['Mean_Accuracy_start']

        # Sorting for consistent order
        plot_data['Sort_Key'] = plot_data[group_by_columns].apply(lambda row: '_'.join(map(str, row)), axis=1)
        plot_data = plot_data.sort_values('Sort_Key')

        # Calculate mean and standard deviation for each slice position
        mean_std_data = network_data.groupby('Slice')['Mean_Accuracy'].agg(['mean', 'std']).reset_index()

        # Plotting parameters
        bar_width = 0.7  # Width of each bar
        group_width = 1.5  # Width of each trio group
        group_spacing = 3  # Spacing between groups

        # Create the plot
        fig, ax = plt.subplots(figsize=(25, 10))

        # Plot
        for i, (_, row) in enumerate(plot_data.iterrows()):
            # Calculate bar positions with spacing
            base_pos = i * (group_width + group_spacing)

            # Plot individual bars
            start_bar = ax.bar(base_pos, row['Mean_Accuracy_start'], bar_width, label='Start' if i == 0 else '',
                               color='#e07a5f',
                               alpha=0.8, yerr=row['SEM_start'], capsize=2)
            middle_bar = ax.bar(base_pos + bar_width, row['Mean_Accuracy_middle'], bar_width,
                                label='Middle' if i == 0 else '',
                                color='#92140c', alpha=0.8, yerr=row['SEM_middle'], capsize=2)
            end_bar = ax.bar(base_pos + 2 * bar_width, row['Mean_Accuracy_end'], bar_width,
                             label='End' if i == 0 else '',
                             color='#1e1e24', alpha=0.8, yerr=row['SEM_end'], capsize=2)

            # Calculate and annotate growth
            growth = row['Growth_Percentage']
            growth_color = 'green' if growth > 0 else 'red'

            # Annotate growth percentage
            if growth > 0:
                # Green arrow above the number
                ax.text(
                    base_pos + bar_width,
                    np.max([row['Mean_Accuracy_start'], row['Mean_Accuracy_middle'], row['Mean_Accuracy_end']]) + 0.01,
                    f'▲ \n{growth:.1f}%',
                    ha='center',
                    color='green',
                    fontweight='bold',
                    fontsize=9
                )
            else:
                # Red arrow below the number
                ax.text(
                    base_pos + bar_width,
                    np.max([row['Mean_Accuracy_start'], row['Mean_Accuracy_middle'], row['Mean_Accuracy_end']]) + 0.01,
                    f'{growth:.1f}%\n ▼',
                    ha='center',
                    color='red',
                    fontweight='bold',
                    fontsize=9
                )

        # Customize the plot
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Mean Accuracy Across Brain Regions - NET: {network} Duration:{config["dur"]} Offset:{config["offset"]}', fontsize=12)

        # Set x-ticks
        x_tick_positions = [i * (group_width + group_spacing) + bar_width for i in range(len(plot_data))]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(
            [x_label_format(row) for _, row in plot_data.iterrows()],
            rotation=90,
            ha='right',
            fontsize=8
        )

        # total mean and std box
        text_lines = []
        for _, row in mean_std_data.iterrows():
            slice_type = row['Slice'].capitalize()
            mean_val = row['mean']
            std_val = row['std']
            text_lines.append(f"{slice_type}: Mean {mean_val:.2f} | Std {std_val:.2f}")
        text_content = "\n".join(text_lines)
        ax.text(
            0.02, 0.95, text_content, fontsize=10, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7)
        )

        # Add legends for bar colors and growth arrows
        bar_legend_handles = [
            Patch(color='#e07a5f', label='Start'),
            Patch(color='#92140c', label='Middle'),
            Patch(color='#1e1e24', label='End')
        ]
        growth_legend_handles = [
            Line2D([0], [0], color='green', marker='^', markersize=10, linestyle='None',
                   label='Positive Growth (End > Start)'),
            Line2D([0], [0], color='red', marker='v', markersize=10, linestyle='None',
                   label='Negative Growth (End < Start)')
        ]
        # Combine legends
        ax.legend(handles=bar_legend_handles + growth_legend_handles, loc='upper right', title="Legend", fontsize=10)

        ax.set_ylim([0, 1])
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"slice_analysis_results_dur{config["dur"]}_offset{config["offset"]}_{config["k_folds"]}fold_rest{config["is_rest"]}_{network}.png"))
        plt.close()

    # ===== Cross Region Analysis ===== #
    #TODO: fix plot x-axis areas names and debug it
    df["Sub_Area"] = df["Sub_Area"].fillna("NSA")
    # Don't fill NaN with anything — treat 'NA' string as just another value
    df_filtered = df[['Hemisphere', 'Network', 'Sub_Area', 'Index', 'Slice', 'Mean_Accuracy']]

    # Pivot to get start and end side by side
    pivot_df = df_filtered.pivot_table(
        index=['Hemisphere', 'Network', 'Sub_Area', 'Index'],
        columns='Slice',
        values='Mean_Accuracy'
    )

    # Drop areas without either start or end
    pivot_df = pivot_df.dropna(subset=['start', 'end'])

    # Compute relative growth
    pivot_df['Growth'] = ((pivot_df['end'] - pivot_df['start']) / pivot_df['start']) * 100

    # Flatten the DataFrame for plotting
    pivot_df = pivot_df.reset_index()

    # Construct a readable area name for x-axis
    pivot_df['Area'] = (
            pivot_df['Hemisphere'] + '_' +
            pivot_df['Network'] + '_' +
            pivot_df['Sub_Area'].astype(str) + '_' +
            pivot_df['Index'].astype(str)
    )

    # Sort by growth descending
    pivot_df = pivot_df.sort_values(by='Growth', ascending=False)

    # Assign colors to Networks
    unique_networks = pivot_df['Network'].unique()
    palette = sns.color_palette("tab10", len(unique_networks))
    network_color_map = dict(zip(unique_networks, palette))

    # Plot
    plt.figure(figsize=(20, 6))
    sns.barplot(
        data=pivot_df,
        x='Area',
        y='Growth',
        hue='Network',
        palette=network_color_map,
        dodge=False
    )

    plt.title("end-start Accuracy Growth per Brain Area", fontsize=16)
    plt.xlabel("Brain Area", fontsize=12)
    plt.ylabel("Accuracy Growth (end-start in %)", fontsize=12)
    plt.ylim([-20, 100])
    ticks = list(range(0, len(df) // 3, 50))
    labels = [str(i) for i in ticks]
    plt.xticks(ticks=ticks, labels=labels, fontsize=10)

    plt.legend(title='Network', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
        f"slice_analysis_results_dur{config["dur"]}_offset{config["offset"]}_{config["k_folds"]}fold_rest{config["is_rest"]}_cross_region.png"))
    plt.close()



def main():
    config = load_config('tools/params_config.json')
    output_dir = os.path.join(config["results_dir"], "Slice_Analysis")
    Path(output_dir).mkdir(exist_ok=True)

    log.info("Running Slice Analysis")
    log.info(f"Input Dir : {config['directory']}")
    log.info(f"Output Dir: {output_dir}")

    df = full_slice_analysis(directory=config["directory"], output_dir=output_dir,config=config, slices=None)
    plot_slice_analysis(config=config, output_dir=output_dir, df=df)

if __name__ == "__main__":
    main()

