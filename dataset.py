import json
import pandas as pd


def Read_Clinc(pathname):
    all_data = []
    all_labels = []
    label_map = {}
    banking = ["transfer", "transactions", "balance", "freeze_account", "pay_bill",
               "bill_balance", "bill_due", "interest_rate", "routing", "min_payment",
               "order_checks", "pin_change", "report_fraud", "account_blocked", "spending_history"]
    credit_cards = ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                    "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
                    "credit_limit_change", "damaged_card", "replacement_card_duration", "improve_credit_score",
                    "expiration_date"]
    kitchen_dining = ["recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
                      "ingredients_list", "ingredient_substitution", "cook_time", "food_last", "meal_suggestion",
                      "restaurant_reservation", "confirm_reservation", "how_busy", "cancel_reservation",
                      "accept_reservations"]
    home = ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist",
            "todo_list", "todo_list_update", "calendar", "calendar_update", "what_song",
            "order", "order_status", "reminder", "reminder_update", "smart_home"]
    auto_commute = ["traffic", "directions", "gas", "gas_type", "distance",
                    "current_location", "mpg", "oil_change_when", "oil_change_how", "jump_start",
                    "uber", "schedule_maintenance", "last_maintenance", "tire_pressure", "tire_change"]
    travel = ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
              "travel_notification", "carry_on", "timezone", "vaccines", "translate",
              "flight_status", "international_visa", "lost_luggage", "plug_type", "exchange_rate"]
    utility = ["time", "alarm", "share_location", "find_phone", "weather",
               "text", "spelling", "make_call", "timer", "date",
               "calculator", "measurement_conversion", "flip_coin", "roll_dice", "definition"]
    work = ["direct_deposit", "pto_request", "taxes", "payday", "w2",
            "pto_balance", "pto_request_status", "next_holiday", "insurance", "insurance_change",
            "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k", "income"]
    small_talk = ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
                  "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you", "what_are_your_hobbies",
                  "do_you_have_pets", "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"]
    meta = ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
            "whisper_mode", "repeat", "no", "yes", "maybe",
            "change_language", "change_accent", "change_volume", "change_speed", "sync_device"]

    all_labels.extend(banking)
    all_labels.extend(credit_cards)
    all_labels.extend(kitchen_dining)
    all_labels.extend(home)
    all_labels.extend(auto_commute)
    all_labels.extend(travel)
    all_labels.extend(utility)
    all_labels.extend(work)
    all_labels.extend(small_talk)
    all_labels.extend(meta)

    for i, label in enumerate(all_labels):
        l = {label: i}
        label_map = {**label_map, **l}

    with open(pathname, 'r', errors='ignor') as infile:
        domain_list = []
        data = json.load(infile)
        for d in data:
            for _, e in enumerate(data[d]):
                if d == "test" or d == "val" or d == "train":
                    if e[1] in banking:
                        domain_list.append("banking")
                        domain_id = 0
                        label_id = banking.index(e[1])
                    elif e[1] in credit_cards:
                        domain_list.append("credit cards")
                        domain_id = 1
                        label_id = credit_cards.index(e[1])
                    elif e[1] in kitchen_dining:
                        domain_list.append("kitchen dining")
                        domain_id = 2
                        label_id = kitchen_dining.index(e[1])
                    elif e[1] in home:
                        domain_list.append("home")
                        domain_id = 3
                        label_id = home.index(e[1])
                    elif e[1] in auto_commute:
                        domain_list.append("auto commute")
                        domain_id = 4
                        label_id = auto_commute.index(e[1])
                    elif e[1] in travel:
                        domain_list.append("travel")
                        domain_id = 5
                        label_id = travel.index(e[1])
                    elif e[1] in utility:
                        domain_list.append("utility")
                        domain_id = 6
                        label_id = utility.index(e[1])
                    elif e[1] in work:
                        domain_list.append("work")
                        domain_id = 7
                        label_id = work.index(e[1])
                    elif e[1] in small_talk:
                        domain_list.append("small talk")
                        domain_id = 8
                        label_id = small_talk.index(e[1])
                    elif e[1] in meta:
                        domain_list.append("meta")
                        domain_id = 9
                        label_id = meta.index(e[1])
                    else:
                        print(e[1])

                    text_raw = e[0]
                    # data_dict = {"label": label_map[e[1]],
                    #              "context": text_raw,
                    #              "domain": domain_id}
                    label_raw = e[1]
                    data_dict = {"label": label_id,
                                 "context": text_raw,
                                 "domain": domain_id}
                    all_data.append(data_dict)

    # sorted_all_data = sorted(all_data, key=lambda i:i['label'])
    # all_example_data = []
    # tmp_example = []
    # tmp_id = 0
    # tmp_label_id = 0
    # for i in range(len(sorted_all_data)):
    #     if tmp_id < 150:
    #         tmp_id += 1
    #         data_dict = {"label": tmp_label_id,
    #                      "context": sorted_all_data[i]['context'],
    #                      "domain": sorted_all_data[i]['domain']}
    #         tmp_example.append(data_dict)
    #     if tmp_id == 150:
    #         tmp_id = 0
    #         tmp_label_id += 1
    #         all_example_data.append(tmp_example)
    #         tmp_example = []
    #
    # for i in range(150):
    #     for j in range(150):
    #         assert (all_example_data[i][0]['label'] == all_example_data[i][j]['label'])
    #         assert (all_example_data[i][0]['domain'] == all_example_data[i][j]['domain'])
    return all_data


def Get_Clinc_domain():
    train_domains = list(range(4))
    val_domains = list(range(4, 5))
    test_domains = list(range(5, 10))
    return train_domains, val_domains, test_domains


def Get_Clinc():
    train_classes = list(range(15))
    val_classes = list(range(15))
    test_classes = list(range(15))
    return train_classes, val_classes, test_classes


def Read_Banking(label_path, data_path1, data_path2):
    all_data1 = pd.read_csv(data_path1, delimiter=",")
    all_data2 = pd.read_csv(data_path2, delimiter=",")
    all_data = []
    label_map = {}
    categories = []

    with open(label_path, 'r', errors='ignor') as infile:
        data = json.load(infile)
        for line in data:
            categories.append(line)
    for i, label in enumerate(categories):
        l = {label: i}
        label_map = {**label_map, **l}

    for i in range(len(categories)):
        example1 = []
        for j in range(len(all_data1)):
            if categories[i] == all_data1['category'][j]:
                text_raw = all_data1['text'][j]
                label_raw = all_data1['category'][j]
                label = label_map[label_raw]
                m = {"label": label, "context": text_raw}
                example1.append(m)
        example2 = []
        for j in range(len(all_data2)):
            if all_data2['category'][j] == categories[i]:
                text_raw = all_data2['text'][j]
                label_raw = all_data2['category'][j]
                label = label_map[label_raw]
                m = {"label": label, "context": text_raw}
                example2.append(m)
        all_data.extend(example1)
        all_data.extend(example2)
    return all_data


def Get_Banking():
    train_classes = list(range(30))
    val_classes = list(range(30, 45))
    test_classes = list(range(45, 77))

    return train_classes, val_classes, test_classes


def Read_HuffPost(pathname):
    data = []
    with open(pathname, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = eval(line)
            data_dic = {"label": line['label'], "context": line['text']}
            data.append(data_dic)
    return data


def Get_HuffPost():

    train_classes = list(range(20))
    val_classes = list(range(20, 25))
    test_classes = list(range(25, 41))

    return train_classes, val_classes, test_classes


def MetaSet_Split(all_data, train_classes, val_classes, test_classes, args):
    train_data, val_data, test_data = [], [], []

    if args.data_name == 'clinc':
        train_domains, val_domains, test_domains = Get_Clinc_domain()
        for data in all_data:
            if data['domain'] in train_domains:
                train_data.append(data)
            if data['domain'] in val_domains:
                val_data.append(data)
            if data['domain'] in test_domains:
                test_data.append(data)
    else:
        for data in all_data:
            if data['label'] in train_classes:
                train_data.append(data)

            if data['label'] in val_classes:
                val_data.append(data)

            if data['label'] in test_classes:
                test_data.append(data)
    return train_data, val_data, test_data
