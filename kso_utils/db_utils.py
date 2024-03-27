# base imports
import sqlite3
import logging
from pathlib import Path
import pandas as pd

# util imports
import kso_utils.db_starter.schema as schema
from kso_utils.project_utils import Project


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# SQL specific functions
def create_connection(db_file: str):
    """create a database connection to the SQLite database
        specified by db_file

    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    db_path = Path(db_file)
    try:
        if not db_path.parent.exists():
            if not db_path.parent == Path(""):
                db_path.parent.mkdir(parents=True)
                db_path.parent.chmod(0o777)
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
        db_path.chmod(0o777)
        return conn
    except sqlite3.Error as e:
        logging.error(e)

    return conn


def drop_table(conn: sqlite3.Connection, table_name: str):
    """
    Safely remove a table from a Sql db

    :param conn: the Connection object
    :param table_name: table of interest
    """
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    try:
        cursor.execute(f"DELETE FROM {table_name}")
    except Exception as e:
        logging.info(f"Table doesn't exist, {e}")
        return
    logging.debug(f"Previous content from the {table_name} table have been cleared.")

    # Commit your changes in the database
    conn.commit()


def insert_many(conn: sqlite3.Connection, data: list, table: str, count: int):
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


def retrieve_query(conn: sqlite3.Connection, query: str):
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
        logging.error(e)

    rows = cur.fetchall()

    return rows


def execute_sql(conn: sqlite3.Connection, sql: str):
    """Execute multiple SQL statements without return

    :param conn: Connection object
    :param sql: a string of SQL statements
    :return:
    """
    try:
        c = conn.cursor()
        c.executescript(sql)
    except sqlite3.Error as e:
        logging.error(e)


def add_to_table(
    conn: sqlite3.Connection, table_name: str, values: list, num_fields: int
):
    """
    This function adds multiple rows of data to a specified table in a SQLite database.

    :param conn: SQL connection object
    :param table_name: The name of the table in the database where the values will be added
    :type table_name: str
    :param values: The `values` parameter is a list of tuples, where each tuple represents a row of data
    to be inserted into the specified table. The number of values in each tuple should match the
    `num_fields` parameter, which specifies the number of columns in the table
    :type values: list
    :param num_fields: The parameter `num_fields` is an integer that represents the number of fields or
    columns in the table where the values will be inserted. This parameter is used to ensure that the
    correct number of values are being inserted into the table
    :type num_fields: int
    """

    try:
        insert_many(
            conn,
            values,
            table_name,
            num_fields,
        )
    except sqlite3.Error as e:
        logging.error(e)

    conn.commit()

    logging.info(f"Updated {table_name} table from the temporary database")


def test_table(df: pd.DataFrame, table_name: str, keys: list = ["id"]):
    """
    The function checks if a given DataFrame has any NULL values in the specified key columns and logs
    an error message if it does.

    :param df: A pandas DataFrame that represents a table in a database
    :type df: pd.DataFrame
    :param table_name: The name of the table being tested, which is a string
    :type table_name: str
    :param keys: The `keys` parameter is a list of column names that are used as keys to uniquely
    identify each row in the DataFrame `df`. The function `test_table` checks that there are no NULL
    values in these key columns, which would indicate that some rows were not properly matched
    :type keys: list
    """
    try:
        # check that there are no id columns with a NULL value, which means that they were not matched
        assert len(df[df[keys].isnull().any(axis=1)]) == 0
    except AssertionError:
        logging.error(
            f"The table {table_name} has invalid entries, please ensure that all columns are non-zero"
        )
        logging.error(f"The invalid entries are {df[df[keys].isnull().any(axis=1)]}")


def get_df_from_db_table(conn: sqlite3.Connection, table_name: str):
    """
    This function connects to a specific table from the sql database
    and returns it as a pd DataFrame.

    :param conn: SQL connection object
    :param table_name: The name of the table you want to get from the database
    :return: A dataframe
    """

    if conn is not None:
        cursor = conn.cursor()
    else:
        return
    # Get column names
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # Get column names
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    columns = [col[1] for col in cursor.fetchall()]

    # Create a DataFrame from the data
    df = pd.DataFrame(rows, columns=columns)

    return df


def get_schema_table_names(conn: sqlite3.Connection):
    """
    > This function retrieves a list with table names of the sql db

    :param conn: SQL connection object
    """

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]

    return table_names


def get_column_names_db(conn: sqlite3.Connection, table_i: str):
    """
    > This function returns the "column" names of the sql table of interest

    :param conn: SQL connection object
    :param table_i: a string of the name of the table of interest
    :return: A list of column names of the table of interest
    """

    # Get the data of the table of interest
    data = conn.execute(f"SELECT * FROM {table_i}")

    # Save in a dictionary the column names of the table
    field_names = {}
    for i in data.description:
        field_names[i[0]] = i[0]

    return field_names


def cols_rename_to_schema(
    project: Project,
    table_name: str,
    df: pd.DataFrame,
    reverse_lookup: bool = False,
):
    """
    > This function renames columns of a df (of one of the three intial project csv files)
    to match the names used in the schema. This deals with csv files having different
    project-specific column names

    :param project: The project object
    :param df: a dataframe with the information of the local csv
    :param table_name: The name of the table in the database where the data is stored
    :param reverse_lookup: a boolean value to reverse the dict if formatting from schema to csv
    """

    # Get the spyfish-specific column names and their correspondent
    # schema fields
    if project.Project_name in ["Spyfish_Aotearoa", "Spyfish_BOPRC"]:
        from kso_utils.spyfish_utils import get_spyfish_col_names

        col_names_lookup = get_spyfish_col_names(table_name)

    # Get the koster-specific column names and their correspondent
    # schema fields
    if project.Project_name == "Koster_Seafloor_Obs":
        from kso_utils.koster_utils import get_koster_col_names

        col_names_lookup = get_koster_col_names(table_name)

    # Rename project-specific columns using the dictionary
    if "col_names_lookup" in locals():
        if reverse_lookup:
            # Reverse the dictionaries if formatting from schema to csv
            col_names_lookup = dict(
                zip(col_names_lookup.values(), col_names_lookup.keys())
            )

        df = df.rename(columns=col_names_lookup)

    return df


# Utility functions for common database operations
def create_db(db_path: str):
    """Create a new database for the project

    :param db_path: path of the database file
    :return:
    """
    db_file = Path(db_path)

    # Delete previous database versions if exists
    if db_file.exists():
        db_file.unlink()

    # Get sql command for db setup
    sql_setup = schema.sql

    # create a database connection
    conn = create_connection(str(db_file))

    # create tables
    if conn is not None:
        # execute sql
        execute_sql(conn, sql_setup)
        return "Database creation success"
    else:
        return "Database creation failure"


def populate_db(
    conn: sqlite3.Connection, project: Project, local_df: pd.DataFrame, init_key=str
):
    """
    > This function processes and tests the initial csv files compatibility with sql db
    and populates the table of interest

    :param conn: SQL connection object
    :param project: The project object
    :param local_df: a dataframe with the information of the local csv to populate from
    :param init_key: a string of the initial key of the local csv and the name of the db table to populate
    """

    # Process the csv of interest and tests for compatibility with sql table
    local_df = process_test_csv(
        conn=conn,
        project=project,
        local_df=local_df,
        init_key=init_key,
    )

    # Only populate the tables if df is not empty
    if not local_df.empty:
        # Add values of the processed csv to the sql table of interest
        add_to_table(
            conn=conn,
            table_name=init_key,
            values=[tuple(i) for i in local_df.values],
            num_fields=len(local_df.columns),
        )


def process_test_csv(
    conn: sqlite3.Connection, project: Project, local_df: pd.DataFrame, init_key=str
):
    """
    > This function process a csv of interest and tests for compatibility with the
    respective sql table of interest

    :param conn: SQL connection object
    :param project: The project object
    :param local_df: a dataframe with the information of the local csv to populate from
    :param init_key: a string corresponding to the name of the initial key of the local csv
    :return: a string of the category of interest and the processed dataframe
    """

    # Rename potential project-specific column names to "standard" schema names
    local_df = cols_rename_to_schema(
        project=project,
        table_name=init_key,
        df=local_df,
    )

    # Set the id of the df of interest
    if init_key == "sites":
        table_id = "site_id"

    elif init_key == "movies":
        table_id = "movie_id"

        from kso_utils.movie_utils import select_project_movies

        # Select only the movies that are relevant to the project
        local_df = select_project_movies(project, local_df)

        # Reference movies with their respective sites
        sites_df = get_df_from_db_table(conn, "sites")[["id", "siteName"]].rename(
            columns={"id": "site_id"}
        )

        # Merge df (aka movies) and sites dfs
        local_df = pd.merge(local_df, sites_df, how="left", on="siteName")

    elif init_key == "species":
        table_id = "species_id"

    elif init_key == "photos":
        table_id = "ID"

    else:
        logging.error(
            f"{init_key} has not been processed because the db schema does not have a table for it"
        )

    # Create a dictionary with the table-specific column id and its schema match
    id_lookup = {table_id: "id"}

    # Rename id columns using the dictionary
    local_df = local_df.rename(columns=id_lookup)

    # Roadblock to ensure cols match schema
    ##################
    # Get the "standard" schema column names of the table of interest
    col_names_dic = get_column_names_db(conn, init_key)

    # Check the column names of the df are standard
    column_names = local_df.columns
    required_columns = col_names_dic.values()

    # Modify the dictionary if the df has different column names
    if not all(col in column_names for col in required_columns):
        missing_cols = [col for col in required_columns if col not in column_names]
        # Log the issue
        logging.error(
            f"{missing_cols} column(s) not found and"
            f" are required for the {init_key}'s schema table"
            f" The col names are:{column_names}"
        )

    # Select only columns that have fields in the sql table
    local_df = local_df[[c for c in required_columns if c in local_df.columns]]

    # Roadblock to prevent empty rows in id_columns
    test_table(local_df, init_key, [local_df.columns[0]])

    return local_df


def check_species_meta(csv_paths: dict, conn: sqlite3.Connection):
    """
    > The function `check_species_meta` loads the csv with species information and checks if it is empty

    :param csv_paths: a dictionary with the paths of the csv files with info to initiate the db
    :param conn: SQL connection object
    """

    # Load the csv with movies information
    species_df = pd.read_csv(csv_paths["local_species_csv"])

    # Retrieve the names of the basic columns in the sql db
    field_names = list(get_column_names_db(conn, "species").values())

    # Select the basic fields for the db check
    df_to_db = species_df[[c for c in species_df.columns if c in field_names]]

    # Roadblock to prevent empty lat/long/datum/countrycode
    test_table(df_to_db, "species", df_to_db.columns)

    logging.info("The species dataframe is complete")


def add_db_info_to_df(
    project: Project,
    conn: sqlite3.Connection,
    csv_paths: dict,
    df: pd.DataFrame,
    table_name: str,
    cols_interest: str = "*",
):
    """
    > This function retrieves information from a sql table and adds it to
    the df

    :param project: The project object
    :param conn: SQL connection object
    :param csv_paths: a dictionary with the paths of the csv files with info to initiate the db
    :param df: a dataframe with the information of the local csv to populate from
    :param table_name: The name of the table in the database where the data is stored
    :param cols_interest: list,
    """
    # Retrieve the sql as a df
    query = f"SELECT {cols_interest} FROM {table_name}"
    sql_df = pd.read_sql_query(query, conn)

    # Set the column to merge dfs on right to "id" as default
    right_on_col = "id"

    # Set movies table
    if table_name == "movies":
        # Add survey information as part of the movie info
        if "local_surveys_csv" in csv_paths.keys():
            from kso_utils.spyfish_utils import add_spyfish_survey_info

            sql_df = add_spyfish_survey_info(sql_df, csv_paths)

        # Save the name of the column to merge dfs on
        left_on_col = "movie_id"

    # Set subjects table
    elif table_name == "subjects":
        # Save the name of the columns to merge dfs on
        left_on_col = "subject_ids"

    # Set sites table
    elif table_name == "sites":
        # Save the name of the columns to merge dfs on
        left_on_col = "site_id"

    # Set species table
    elif table_name == "species":
        # Save the name of the columns to merge dfs on
        left_on_col = "commonName"
        right_on_col = "commonName"

        if "label" in df.columns:
            df[right_on_col] = df["label"]

        from kso_utils.zooniverse_utils import clean_label

        # Match format of species name to Zooniverse labels
        sql_df[right_on_col] = sql_df[right_on_col].apply(clean_label)
        df[left_on_col] = df[left_on_col].apply(clean_label)

    else:
        logging.error(
            f"The table_name specified ({table_name}) doesn't have a merging option"
        )

    # Ensure id columns that are going to be used to merge are int
    if "id" in left_on_col:
        # Ensure there are no NaN values found in the column id column
        if df[left_on_col].isna().any() or (df[left_on_col] == "None").any():
            logging.error(
                f"Error: NaN values found in the {left_on_col} column of {table_name}."
            )

        else:
            df[left_on_col] = df[left_on_col].astype(float).astype(int)

    # Combine the original and sqldf dfs
    comb_df = pd.merge(
        df, sql_df, how="left", left_on=left_on_col, right_on=right_on_col
    )

    # Check for rows with NaN values in the merged DataFrame
    missing_values = comb_df[right_on_col].isnull()

    # If there are missing values, raise an issue
    if missing_values.any():
        # Log a warning or raise an exception with relevant information
        logging.error(
            f"Some rows in df do not have corresponding values in sql_df. Rows with missing values are: {comb_df[missing_values]}"
        )

    # Drop the id column to prevent duplicated column issues
    comb_df = comb_df.drop(columns=["id"], errors="ignore")

    return comb_df


# Function to match species selected to species id
def get_species_ids(conn: sqlite3.Connection, species_list: list):
    """
    # Get ids of species of interest
    """
    if len(species_list) == 1:
        species_ids = pd.read_sql_query(
            f'SELECT id FROM species WHERE commonName=="{species_list[0]}"', conn
        )["id"].tolist()
    else:
        species_ids = pd.read_sql_query(
            f"SELECT id FROM species WHERE commonName IN {tuple(species_list)}", conn
        )["id"].tolist()

    return species_ids
