# base imports
import sqlite3
import logging
from pathlib import Path
from importlib import import_module
import pandas as pd

# util imports
import kso_utils.db_starter.schema as schema
from kso_utils.project_utils import Project


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# SQL specific functions
def empty_table(conn: sqlite3.Connection, table_name: str):
    """
    Safely empty a table from its data in a Sql db

    :param conn: the Connection object
    :param table_name: table of interest
    """
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    try:
        cursor.execute(f"DELETE FROM {table_name}")
        logging.info(
            f"Previous content from the '{table_name}' table have been cleared."
        )
        conn.commit()  # Commit your changes in the database
    except sqlite3.Error:
        cursor.execute("PRAGMA table_list;")
        raise sqlite3.Error(
            f"Table '{table_name}' does not exist, only tables {[row[1] for row in cursor.fetchall()]}"
        )


def add_to_table(conn: sqlite3.Connection, table_name: str, values: list):
    """
    This function adds multiple rows of data to a specified table in a SQLite database.

    :param conn: SQL connection object
    :param table_name: The name of the table in the database where the values will be added
    :type table_name: str
    :param values: The `values` parameter is a list of tuples, where each tuple represents
    a row of data to be inserted into the specified table. The lenght of the tuples should
    be equal to the amount of columns in the table.
    :type values: list
    """
    
    # l = [i[0] for i in values]
    # print(values[0])
    # print(len(l), len(set(l)))
    # print({', '.join(['?' for _ in values[0]])})

    # Check if 'id' column exists in sites

    
    # # Check site_id values in movies that are not in sites
    # invalid_sites = local_df[~local_df["site_id"].isin(site_ids)]
    # print(invalid_sites)

    try:
        cur = conn.cursor()
        cur.executemany(
            f"INSERT INTO {table_name} VALUES ({', '.join(['?' for _ in values[0]])})",
            values,
        )
        conn.commit()

    except sqlite3.Error as e:
        # Check if the error code indicates a foreign key constraint violation
        print(e)
        if e.args[0].startswith("FOREIGN KEY constraint failed") :
            # Get the table of the problematic value
            df1 = get_df_from_db_table(conn, table_name)

            # Get the last row added to the table
            value_i = len(df1)

            # Save the problematic value
            foreign_key_value = values[value_i]
            raise sqlite3.DataError(
                f"Foreign Key Constraint Error (table: {table_name}), Error values: {foreign_key_value}, Full Error: {e} "
            )
        else:
            raise sqlite3.Error(f"add_to_table failed: {e}")

    logging.info(f"Updated {table_name} table from the temporary database")


def test_table_for_none(df: pd.DataFrame, table_name: str, keys: list = ["id"]):
    """
    The function checks if a given DataFrame has any NULL values in the specified key columns and logs
    an error message if it does.

    :param df: A pandas DataFrame that represents a table in a database
    :type df: pd.DataFrame
    :param table_name: The name of the table being tested, which is a string
    :type table_name: str
    :param keys: The `keys` parameter is a list of column names that are used as keys to uniquely
    identify each row in the DataFrame `df`. The function `test_table_for_none` checks that there are no NULL
    values in these key columns, which would indicate that some rows were not properly matched
    :type keys: list
    """
    assert (
        len(df[df[keys].isnull().any(axis=1)]) == 0
    ), f"The table {table_name} has invalid entries, please ensure that all columns are non-zero. The invalid entries are {df[df[keys].isnull().any(axis=1)]}"


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
    cursor.execute("SELECT name FROM sqlite_master WHERE type=\'table\'")
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
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas Dataframe for df, got {type(df)} instead.")

    # Check if there is a function to rename the columns
    try:
        mod = import_module(project.utils_path)
        func_get_col_names = getattr(mod, "get_col_names")
        col_names_lookup = func_get_col_names(table_name)
        logging.info(
            f"{project.Project_name} has colums to rename, these are successfully retrieved."
        )
    except ModuleNotFoundError as e:
        if e.name == project.utils_path:
            logging.info(
                f"{project.Project_name} does not define a utils module for renaming. Skipping renaming."
            )
            return df
        else:
            raise  # re-raise unexpected import errors
    except AttributeError as e:
        logging.info(
            f"{project.Project_name} does not define a 'get_col_names' function. Skipping renaming."
        )
        return df

    if not isinstance(col_names_lookup, dict):
        raise TypeError(
            f"The get_col_names function needs to return a dict, got a {type(col_names_lookup)}"
        )

    # Actually rename the columns
    if reverse_lookup:
        # Reverse the dictionaries if formatting from schema to csv
        col_names_lookup = dict(zip(col_names_lookup.values(), col_names_lookup.keys()))
    df = df.rename(columns=col_names_lookup)
    logging.info("The columns are successfully renamed.")
    return df


# Utility functions for common database operations
def create_db(db_path_str: str) -> sqlite3.Connection:
    """Create a new database for the project

    :param db_path: str of the path of the database file
    :return:
    """
    db_path = Path(db_path_str)
    # Delete previous database versions if exists
    if db_path.exists():
        db_path.unlink()

    # Get sql command for db setup
    sql_setup = schema.sql
    # create the DB file if it does not exist
    if not db_path.exists():
        if not db_path.parent == Path(""):
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.parent.chmod(0o777)
        with open(db_path, "w") as f:
            pass
    db_path.chmod(0o777)
    # And connect to it
    try:
        conn = sqlite3.connect(db_path_str)
        conn.execute("PRAGMA foreign_keys = 1")
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to connect to {db_path_str}: {e}")

    try:
        c = conn.cursor()
        c.executescript(sql_setup)
    except sqlite3.Error as e:
        raise sqlite3.Error(
            f"Failed to init DB schema: {e}. SQL statements: {sql_setup}"
        )
    logging.info("Database creation success")
    return conn


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

    print(init_key, "yo111", local_df.columns)
    if 'id' in local_df.columns:
        duplicates = local_df[local_df.duplicated('id', keep=False)]
        if not duplicates.empty:
            print("Duplicate id entries:")
            print(duplicates)
        local_df = local_df.drop_duplicates(subset='id', keep='first')
    # Only populate the tables if df is not empty
    if not local_df.empty:
        # Add values of the processed csv to the sql table of interest
        add_to_table(
            conn=conn,
            table_name=init_key,
            values=[tuple(i) for i in local_df.values],
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

    # Map the init_key to the table_id from the csv
    table_id = {
        "sites": "site_id",
        "movies": "movie_id",
        "species": "species_id",
        "photos": "ID",
    }
    assert (
        init_key in table_id.keys()
    ), f"init_key should be one of {table_id.keys()} so that the db has a table for it, but instead got {init_key}"

    # For Spyfish_Aotearoa: Select only movies that are a good deployment / have good visibility
    if init_key == "movies":
        if project.Project_name in ["Spyfish_Aotearoa"]:
            assert (
                "IsBadDeployment" in local_df
            ), "The movies csv from Spyfish_Aotearoa expects to have a column called IsBadDeployment, but it is missing."
            # Check for missing values in IsBadDeployment column
            if local_df["IsBadDeployment"].isnull().any():
                raise ValueError(
                    "The 'IsBadDeployment' column contains missing values. Please handle missing values before proceeding."
                )
            else:
                local_df = local_df.loc[~local_df.IsBadDeployment].drop(
                    "IsBadDeployment", axis=1
                )



    if init_key=="movies":
        
        local_df = local_df.drop(columns=['DepthStrata', 'Created By', 'NotesDeployment', 
       'SurveyID',  'EventTimeEnd',
       'ReplicateWithinSite', 'EventTimeStart',  'NZMHCS_Abiotic', 'Weather', 'TideLevel',
        'DepthDeployment', 'UnderwaterVisibility','DropID',
       'NZMHCS_Biotic'] , errors='ignore')
        # Step 1: Read valid site_ids from the sites table
        site_ids = pd.read_sql_query("SELECT siteName FROM sites", conn)
        print(site_ids.columns)
        site_ids=  site_ids["siteName"].tolist()
        
        # Step 2: Filter local_df to only include rows with site_id in sites table
        local_df = local_df[local_df["siteName"].isin(site_ids)]
        local_df = local_df.dropna(subset=["fileName"])

    # Rename id columns to simply "id"
    local_df = local_df.rename(columns={table_id[init_key]: "id", 
                                        # 'Targeted':  'kingdom', 
                                        'kingdom': 'Targeted', 
                                        # 'aphiaID2' : 'DOC_TaxonID'
                                        # 'DOC_TaxonID':'aphiaID'
                                        'fileName': 'filename',
                                        # 'DropID' : 'id'
                                       })

    local_df = local_df.drop(columns=[ 'SiteExposure','ControlToMR02',
       'SiteCode', 'ProtectionStatus', 'ControlToMR03',  'ControlToMR01',
       'ProtectionStatusDetails', 'IsControlSite', 'Region',
       'LinkToMarineReserve', 'SiteNameOld' ], errors='ignore')
    
    # Ensure cols of df match db schema
    schema_col_names = get_column_names_db(conn, init_key).values()
    assert set(local_df.columns) == set(
        schema_col_names
    ), f"csv columns and db table columns for {init_key} do not match. The df contains {local_df.columns} and the sql table requires {schema_col_names}. Make sure they contain the same columns."

    # Roadblock to prevent empty rows in id_columns
    test_table_for_none(local_df, init_key, [local_df.columns[0]])

    # Reorder the colums so they are in the order of schema
    # Otherwise the data will be added to the wrong key
    local_df = local_df[[c for c in schema_col_names]]

    
    # print(local_df.columns)
    # print(schema_col_names)


    if init_key=="movies" or init_key=="sites":
        print(local_df.sample(5))

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
    test_table_for_none(df_to_db, "species", df_to_db.columns)

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
        if "site_id" in df.columns:
            left_on_col = "site_id"
        elif "siteName" in df.columns:
            left_on_col = "siteName"
            right_on_col = "siteName"
        else:
            logging.error("No suitable site column found for merge.")

    # Set species table
    elif table_name == "species":
        # Save the name of the columns to merge dfs on
        left_on_col = "commonName"
        right_on_col = "commonName"

        if "label" in df.columns:
            df[right_on_col] = df["label"]

        from kso_upload_copy.notebooks.classify.hold.zooniverse_utils import clean_label

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