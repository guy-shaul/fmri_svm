import os
from pathlib import Path
from data_loader.utils import load_config
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shura

log = shura.get_logger(name= "Movies Duration Plotting (Tool)", level="DEBUG", to_file= False, filename="movies_dur_plot.log", file_format="log")


def plot_clip_durations(data_replica_path, output_path, subject_id=100610):

    try:
        try:
            # Load the CSV
            df = pd.read_csv(data_replica_path)
            log.info(f"Successfully read data from {data_replica_path}")
        except FileNotFoundError:
            log.error(f"File not found: {data_replica_path}")
            raise

        df_subject = df[df['Subject'] == subject_id]
        df_subject = df_subject[df_subject['is_rest'] == 0] # Remove rest periods
        df_subject['y'] = df_subject['y'].astype(int)

        movie_durations = df_subject.groupby('y')['timepoint'].count() # Count duration per movie
        movie_durations = movie_durations.sort_index()

        avg_duration = movie_durations.mean() # Calculate average duration

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(movie_durations.index.astype(str), movie_durations.values, color='steelblue')

        # Add numbers above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, str(height),
                    ha='center', va='bottom', fontsize=10)

        # Dashed line at shortest duration
        ax.axhline(y=181, color='red', linestyle='--', linewidth=1.5)

        # Custom legend handles
        legend_lines = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Shortest Clip Duration (TR)'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=0, marker='',
                   label=f'Avg Duration: {avg_duration:.1f} TR')
        ]

        # Show legend in upper-right outside the plot
        ax.legend(handles=legend_lines, loc='upper left', bbox_to_anchor=(1.01, 1.0))

        ax.set_xlabel('Clip Number')
        ax.set_ylabel('Duration (TR)')
        ax.set_title('Duration Per Clip')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout to make space for legend
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        log.info(f"Successfully plotted movies durations and saved to {output_path}")

    except Exception as e:
        log.error(f"Unexpected error in plot_clip_durations: {type(e).__name__}: {e}")
        raise


if __name__ == '__main__':
    config = load_config('tools/params_config.json')
    output_dir = os.path.join(config["results_dir"], "Duration_Analysis")
    Path(output_dir).mkdir(exist_ok=True)

    log.info("Plotting Movies Duration")
    log.info(f"Input Dir : {config['directory']}")
    log.info(f"Output Dir: {output_dir}")


    plot_clip_durations(
        data_replica_path=os.path.join(config['directory'], 'brain_data_replica.csv'),
        output_path=os.path.join(config['results_dir'], 'Duration_Analysis/movies_durations.png'),
        subject_id=100610
    )
