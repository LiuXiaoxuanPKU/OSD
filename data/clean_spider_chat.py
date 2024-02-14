from collector import Collector

import sqlite3

file_path = "spider_database/train_spider.json"

def transform(i, case, need_label=False):
    
    SQL_prompt = "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "
    
    case["id"] = f"identity_{i}"
    
    db_name = case['db_id']
    db_path = f"spider_database/database/{db_name}/{db_name}.sqlite"
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
    curr_table = cursor.fetchall()

    table_rows = {}
    for table in curr_table:
        table_name = str(table[0])

        cursor_t = con.execute(f"SELECT * from {table_name}")
        names = list(map(lambda x: x[0], cursor_t.description))
        table_rows[table_name] = names
        cursor_t.close()

    cursor.close()
    con.close()

    database_info = "The SQL database has "
    for k, v in table_rows.items():
        database_info = database_info + f"table named {k} with columns {v}, "

    prompt = SQL_prompt + database_info + "Question: "
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content": prompt + case['question']
            },
            {
                "role": "assistant",
                "content": " ".join(case['query_toks_no_value'])
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": prompt + case['question']
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "spider"
    c = Collector(data_name)
    c.collect("train", transform, True)
    c.collect("validation", transform, True)
