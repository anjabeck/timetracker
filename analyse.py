# https://github.com/kevinsblake/NatParksPalettes/blob/main/R/NatParksPalettes.R

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import tqdm
import logging
import argparse
logging.getLogger('matplotlib.font_manager').disabled = True

bryce_canyon_colours = [
    "#882314", "#C0532B", "#CF932C", "#674D53", "#8C86A0", "#724438", "#D5AB85",
    "#3F4330", "#8E7E3C", "#521E0F", "#9C593E", "#DDA569", "#2A4866", "#6592B0"
]


parser = argparse.ArgumentParser(description='Analyse time tracking data.')
parser.add_argument('--maxweeks', type=int, default=52, help='Maximum number of weeks to process')
args = parser.parse_args()


def smooth_box(x, loc, scale):
    flat_edge = scale/2
    func_edge = scale*1.5
    flat_top = np.abs(x - loc) <= flat_edge
    right_edge = (flat_edge <= x - loc) & (x - loc <= func_edge)
    left_edge = (-func_edge <= x - loc) & (x - loc <= -flat_edge)
    f = np.where(flat_top, 2, 0)
    f = np.where(left_edge, (1 + np.cos(np.pi * (x - loc + flat_edge) / scale)), f)
    f = np.where(right_edge, (1 + np.cos(np.pi * (x - loc - flat_edge) / scale)), f)
    return f / (2 * 2 * scale)


def test_smooth_box():
    x = np.linspace(-1, 3, 1000)
    y = smooth_box(x, loc=1, scale=0.5)
    plt.plot(x, np.ones_like(x) * 2)
    plt.plot(x, 1 + np.cos(np.pi * (x-1) / 0.5))
    plt.plot(x, y)
    plt.axvline(1/4, color='gray', linestyle='--')
    plt.axvline(-1/4, color='gray', linestyle='--')
    plt.title('Smooth Box Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig('smooth_box_example.pdf', bbox_inches='tight')
    plt.close()


def process_data():
    dtypes = {
        'Activity': str,
        'Category': str,
        'Category 2': str,
    }
    sheet_id = '1lMJMldi_Y6b7e8hwN8I_TzUCNd-gGFgTbQO-aszVGoQ'

    df = None
    found_weeks = []
    pbar = tqdm.tqdm(total=args.maxweeks, desc='Loading weeks')
    for week in range(1, args.maxweeks + 1):
        pbar.update(1)
        try:
            dfweek = pd.read_excel(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx", sheet_name=f'Week {week}', dtype=dtypes, engine='openpyxl')
        except Exception:
            continue
        found_weeks.append(week)
        if week == 1:
            df = dfweek
        else:
            df = pd.concat([df, dfweek], ignore_index=True)
    pbar.close()
    print(f"Found data for weeks: {found_weeks}")

    # Add date to start and end times for easier time calculations
    df['Duration'] = pd.to_datetime(df['Duration'], format='%H:%M:%S', errors='coerce').dt.time
    df['Start'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Start'].astype(str), errors='coerce')
    df['End'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['End'].astype(str), errors='coerce')

    # Check that durations match start and end times
    pbar = tqdm.tqdm(total=len(df), desc='Checking durations')
    allgood = True
    for _, row in df.iterrows():
        start = row['Start']
        end = row['End']
        duration = row['Duration']
        if pd.isna(start) or pd.isna(end) or pd.isna(duration):
            continue
        calculated_duration = end - start
        if abs(calculated_duration.total_seconds() - (duration.hour * 3600 + duration.minute * 60 + duration.second)) > 60:
            print("Duration mismatch!")
            print(row)
            allgood = False
        pbar.update(1)
    pbar.close()
    if allgood:
        print("Data is consistent.")
    else:
        exit("Found duration mismatches. Exiting.")

    # Remove trailing spaces from Activity names
    df['Activity'] = df['Activity'].str.rstrip()
    df['Category'] = df['Category'].str.rstrip()
    df['Category 2'] = df['Category 2'].str.rstrip()

    return df


def plot_total(df, norm, label, out_name):
    pbar = tqdm.tqdm(total=len(df['Category'].unique()), desc='Plotting totals')
    i = 0
    categories = {}
    for cat in df['Category'].unique():
        dfcat = df[df['Category'] == cat]
        total_duration = pd.to_timedelta(0)
        for dur in dfcat['Duration'].dropna():
            total_duration += pd.to_timedelta(f"{dur.hour}:{dur.minute}:{dur.second}")
        i += 1
        pbar.update(1)
        categories[cat] = total_duration.total_seconds() / norm
    pbar.close()
    # Sort bars by height
    sorted_cats = sorted(categories, key=categories.get, reverse=True)
    sorted_heights = [categories[cat] for cat in sorted_cats]
    sorted_colors = [bryce_canyon_colours[i % len(bryce_canyon_colours)] for i in range(len(sorted_cats))]
    plt.bar(range(len(sorted_cats)), sorted_heights, color=sorted_colors)
    plt.xticks(range(len(sorted_cats)), sorted_cats, rotation=90)
    for i, cat in enumerate(sorted_cats):
        height = categories[cat]
        if height > 0.2 * plt.ylim()[1]:
            plt.text(i, 0.02*plt.ylim()[1], cat, ha='center', va='bottom', fontsize=10, rotation=90, color='white')
        else:
            plt.text(i, height+0.02*plt.ylim()[1], cat, ha='center', va='bottom', fontsize=10, rotation=90, color=sorted_colors[i])
    plt.xticks([])
    plt.ylabel(label)
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()
    return {cat: height for cat, height in zip(sorted_cats, sorted_heights)}


def plot_stacked_daily(df, order=None):
    pbar = tqdm.tqdm(total=len(df['Category'].unique()), desc='Plotting daily activity')
    i = 0
    ybaseline = np.zeros(1000)
    x = np.linspace(0, 24, 1000)
    if order is not None:
        categories = order.keys()
    else:
        categories = df['Category'].unique()
    for cat in categories:
        dfcat = df[df['Category'] == cat]
        y = np.zeros_like(x)
        for start, end in zip(dfcat['Start'].dropna(), dfcat['End'].dropna()):
            start_dt = pd.to_datetime(f"2024-01-01 {start.hour}:{start.minute}:{start.second}")
            end_dt = pd.to_datetime(f"2024-01-01 {end.hour}:{end.minute}:{end.second}")
            timespan = (end_dt - start_dt).total_seconds() / 60 / 5  # in 5-minute bins
            if timespan <= 0:
                continue
            for j in range(int(timespan)):
                time_point = start_dt + pd.to_timedelta(j * 5, unit='minutes')
                x_val = time_point.hour + time_point.minute / 60 + time_point.second / 3600
                y += smooth_box(x, loc=x_val, scale=1/12) * (1 / timespan)
                y += smooth_box(x, loc=x_val+24, scale=1/12) * (1 / timespan)
                y += smooth_box(x, loc=x_val-24, scale=1/12) * (1 / timespan)
        plt.fill_between(x, ybaseline, ybaseline + y, label=cat, color=bryce_canyon_colours[i % len(bryce_canyon_colours)], linewidth=0)
        ybaseline += y
        i += 1
        pbar.update(1)
    pbar.close()
    plt.xticks(np.arange(0, 25, 2))
    plt.xlabel('Time of Day')
    plt.yticks([])
    plt.xlim(0, 24)
    plt.ylim(0, None)
    # Invert legend order to match stacking
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    plt.savefig('activity_times_day.pdf', bbox_inches='tight')
    plt.close()


def plot_stacked_weekly(df, order=None):
    pbar = tqdm.tqdm(total=len(df['Category'].unique()), desc='Plotting weekly activity')
    i = 0
    ybaseline = np.zeros(1000)
    x = np.linspace(0.5, 7.5, 1000)
    weeks = len(df['Date'].dt.isocalendar().week.unique())
    if order is not None:
        categories = order.keys()
    else:
        categories = df['Category'].unique()
    for cat in categories:
        dfcat = df[df['Category'] == cat]
        y = np.zeros_like(x)
        for date, duration in zip(dfcat['Date'].dropna(), dfcat['Duration'].dropna()):
            day = date.isoweekday()
            dur_hours = duration.hour + duration.minute / 60 + duration.second / 3600
            # y[day - 1] += dur_hours
            # y += scipy.stats.norm.pdf(x, loc=day, scale=.5) * dur_hours / weeks
            # y += scipy.stats.norm.pdf(x, loc=day+7, scale=.5) * dur_hours / weeks
            # y += scipy.stats.norm.pdf(x, loc=day-7, scale=.5) * dur_hours / weeks
            y += smooth_box(x, loc=day, scale=0.5) * dur_hours / weeks
            y += smooth_box(x, loc=day+7, scale=0.5) * dur_hours / weeks
            y += smooth_box(x, loc=day-7, scale=0.5) * dur_hours / weeks
        plt.fill_between(x, ybaseline, ybaseline + y, label=cat, color=bryce_canyon_colours[i % len(bryce_canyon_colours)], linewidth=0)
        ybaseline += y
        i += 1
        pbar.update(1)
    pbar.close()
    plt.xticks(np.arange(0.5, 7.5), [])
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in range(1, 8):
        plt.text(day, -0.05, days[day-1], ha='center', va='top', fontsize=8)
    # plt.xlabel('Day of the week')
    plt.ylabel('Average number of hours per day')
    plt.xlim(0.5, 7.5)
    plt.ylim(0, None)
    plt.grid()
    plt.savefig('activity_times_week.pdf', bbox_inches='tight')
    plt.close()


def plot_reading_wpm(df):
    books = df[df['Category'] == 'Reading']['Activity'].unique()

    books_db = pd.read_csv('books.csv', sep='\t')
    books_db.dropna(inplace=True)
    books_db['Rating'] = books_db['Rating'].astype(int)
    books_db['Words'] = books_db['Words'].astype(int)

    plt.xkcd()
    pbar = tqdm.tqdm(total=len(books), desc='Plotting reading WPM')
    i = 0
    wpms = []
    for book in books:
        if book not in books_db['Title'].values:
            exit(f"Word count for book '{book}' not found.")
        this_book = books_db[books_db['Title'] == book]
        img = mpimg.imread(f"book-covers/{book.replace(' ', '-').lower()}.jpg")
        color = img[5, 5, :3] / 255
        dfbook = df[(df['Category'] == 'Reading') & (df['Activity'] == book)]
        total_words = this_book['Words'].values[0]
        total_duration = pd.to_timedelta(0)
        for dur in dfbook['Duration'].dropna():
            total_duration += pd.to_timedelta(f"{dur.hour}:{dur.minute}:{dur.second}")
        total_minutes = total_duration.total_seconds() / 60
        wpm = total_words / total_minutes if total_minutes > 0 else 0
        if wpm > 300:
            plt.bar(i, 300, color=color, width=0.9, hatch='xxx', edgecolor='white')
            plt.bar(i, 200, color=color, width=0.9)
        else:
            plt.bar(i, wpm, color=color, width=0.9)
        stepsize = 0.15
        for s in range(0, this_book['Rating'].values[0]):
            loc = i - (this_book['Rating'].values[0]/2)*stepsize + (s+0.5)*stepsize
            plt.plot(loc, wpm+10, marker='*', color='gold', markersize=12, markeredgecolor='black')
        ax = plt.gca()
        axin = ax.inset_axes([i-0.4, 0, 0.8, 115], transform=ax.transData)
        axin.imshow(img, interpolation='none')
        axin.axis('off')
        i += 1
        wpms.append(wpm)
        pbar.update(1)
    pbar.close()
    plt.xticks([])
    plt.ylim(0, min(np.max(wpms) * 1.2, 275))
    plt.title('Words per Minute')
    plt.savefig('reading_wpm.pdf', bbox_inches='tight')
    plt.close()


test_smooth_box()

df = process_data()
ordered = plot_total(df, norm=3600,
                     label='Total number of hours',
                     out_name='category_durations.pdf')
plot_total(df, norm=60*((df['Date'].max() - df['Date'].min()).days + 1),
           label='Average number of minutes per day',
           out_name='category_durations_dailyaverage.pdf')

plot_stacked_daily(df, order=ordered)
plot_stacked_weekly(df, order=ordered)
plot_reading_wpm(df)
