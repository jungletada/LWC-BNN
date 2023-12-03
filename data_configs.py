data_path = 'data-slim/'
stat_save_path = 'data-slim/statistics/'

dates_selection = [
    "20121017",
    "20121018",
    "20121022",
    "20121023",
    "20121024",
    "20121028",
    "20121030",
    "20121031",
    "20121101",
    "20121117",
    "20121121",
    "20130214",
    "20130215",
    "20130218",
    "20130405",
    "20130406",
    "20130407",
    "20130514",
    "20130515",
]

new_dates = [
    "20121102",
    "20121103",
    "20121104",
    "20121105",
    "20121106",
    "20121107",
    "20121113",
    "20121114",
    "20121115",
    "20121116",
    "20121118",
    "20121119",
    "20121120",
    "20121122",
    "20121123",
]

all_dates = dates_selection

length_dates = len(all_dates)
split_idx = int(length_dates * 0.9)

train_date_selection = all_dates[:split_idx]
test_date_selection = all_dates[split_idx:]

# print(f"Total number of dates: {length_dates}")
# print(f"Total train dates: {split_idx}")
# print(f"Total test dates: {length_dates - split_idx}")
# print(f"Trainset: {train_date_selection}")
# print(f"Testset: {test_date_selection}")