import sqlite3
import requests
import pandas as pd
import numpy as np
import io

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn

def insert_many(conn, data, table, count):
    """
    Insert multiple rows into table
    :param conn: the Connection object
    :param data: data to be inserted into table
    :param table: table of interest
    :param count: number of fields
    :return:
    """

    values = (1,) * count
    values = str(values).replace("1", "?")

    cur = conn.cursor()
    cur.executemany(f"INSERT INTO {table} VALUES {values}", data)

def retrieve_query(conn, query):
    """
    Execute SQL query and returns output
    :param conn: the Connection object
    :param query: a SQL query
    :return:
    """
    try:
        cur = conn.cursor()
        cur.execute(query)
    except sqlite3.Error as e:
        print(e)

    rows = cur.fetchall()

    return rows

def execute_sql(conn, sql):
    """ Execute multiple SQL statements without return
    :param conn: Connection object
    :param sql: a string of SQL statements
    :return:
    """
    try:
        c = conn.cursor()
        c.executescript(sql)
    except sqlite3.Error as e:
        print(e)

def add_to_table(db_path, table_name, values, num_fields):

    conn = create_connection(db_path)

    try:
        insert_many(
            conn, values, table_name, num_fields,
        )
    except sqlite3.Error as e:
        print(e)

    conn.commit()

    print(f"Updated {table_name}")

def test_table(df, table_name, keys=["id"]):
    try:
        # check that there are no id columns with a NULL value, which means that they were not matched
        assert len(df[df[keys].isnull().any(axis=1)]) == 0
    except AssertionError:
        print(
            f"The table {table_name} has invalid entries, please ensure that all columns are non-zero"
        )

def get_id(field_name, table_name, conn, conditions={"a": "=b"}):

    # Get id from a table where a condition is met

    if isinstance(conditions, dict):
        condition_string = f" AND ".join(
            [k + v[0] + f"{v[1:]}" for k, v in conditions.items()]
        )
    else:
        raise ValueError("Conditions should be specified as a dict, e.g. {'a', '=b'}")

    try:
        id_value = retrieve_query(
            conn, f"SELECT {field_name} FROM {table_name} WHERE {condition_string}"
        )[0][0]
    except IndexError:
        id_value = None
    return id_value

def unswedify(string):
    """ Convert ä and ö to utf-8
    """
    if b"\xc3\xa4" in string.encode("utf-8") or b"\xc3\xb6" in string.encode("utf-8"):
        return (
            string.encode("utf-8")
            .replace(b"\xc3\xa4", b"a\xcc\x88")
            .replace(b"\xc3\xb6", b"o\xcc\x88")
            .decode("utf-8")
        )
    elif b"a\xcc\x88" in string.encode("utf-8") or b"o\xcc\x88" in string.encode("utf-8"):
        return (
            string.encode("utf-8")
            .replace(b"a\xcc\x88", b"\xc3\xa4")
            .replace(b"o\xcc\x88", b"\xc3\xb6")
            .decode("utf-8")
        )
    else:
        return string

def clean_species_name(string):
    """Remove whitespace and non-alphanumeric characters from species string"""
    string = string.replace(" ", "_")
    string = ''.join([x.lower() for x in string if x.isalpha() or x in ["_"]])
    return string

def download_csv_from_google_drive(file_url):

    # Download the csv files stored in Google Drive with initial information about
    # the movies and the species

    file_id = file_url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
    url = requests.get(dwn_url).text.encode("ISO-8859-1").decode()
    csv_raw = io.StringIO(url)
    dfs = pd.read_csv(csv_raw)
    return dfs

def find_duplicated_clips(conn):
    
    # Retrieve the information of all the clips uploaded
    subjects_df = pd.read_sql_query(
        f"SELECT id, movie_id, clip_start_time, clip_end_time FROM subjects WHERE subject_type='clip'",
        conn,
    )

    # Find clips uploaded more than once
    duplicated_subjects_df = subjects_df[subjects_df.duplicated(['movie_id', 'clip_start_time','clip_end_time'],
                                                                keep=False)]
    
    # Count how many time each clip has been uploaded
    times_uploaded_df = duplicated_subjects_df.groupby(['movie_id', 'clip_start_time'],
                                              as_index=False).size().to_frame('times')
    
    return times_uploaded_df['times'].value_counts()

# Function to combine classifications received on duplicated subjects
def combine_duplicates(annot_df, duplicates_file_id):
    
    # Download the csv with information about duplicated subjects
    dups_df = download_csv_from_google_drive(duplicates_file_id)
    
    # Include a column with unique ids for duplicated subjects 
    annot_df = pd.merge(annot_df, dups_df, how="left", left_on="subject_ids", right_on="dupl_subject_id")
    
    # Replace the id of duplicated subjects for the id of the first subject
    annot_df['subject_ids'] = np.where(annot_df.single_subject_id.isnull(), annot_df.subject_ids, annot_df.single_subject_id)
    
    return annot_df
