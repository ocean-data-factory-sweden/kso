from pathlib import Path
import sqlite3
import pytest
import pandas as pd

import kso_utils.db_utils as db_utils
import kso_utils.project_utils as project_utils


def test_create_db():
    db_path_str = "sample.db"
    conn = db_utils.create_db(db_path_str)
    assert Path(db_path_str).exists(), f"the db was not created: {db_path_str}"
    assert isinstance(
        conn, sqlite3.Connection
    ), f"Connection is not a sqlite3.Connection: {type(conn)}"
    # To test that the schema was inserted we should fetch information. Example:
    # https://www.tutorialspoint.com/check-if-a-table-exists-in-sqlite-using-python


@pytest.fixture
def connection():
    # db_path is from the template project, defined in "db_starter/projects_list.csv"
    db_path_str = "template_project.db"
    conn = db_utils.create_db(db_path_str)

    yield conn

    conn.close()
    Path(db_path_str).unlink()


def test_get_schema_table_names(connection):
    names = db_utils.get_schema_table_names(connection)
    print(names)
    assert isinstance(
        names, list
    ), f"get_schema_table_names should return a list, instead it returns a {type(names)}"
    # The tables in the db from the template connection are defined in 'kso_utils/db_starter/schema.py'
    # And an extra table 'sqlite_sequence', which is created due to that we use AUTOINCREMENT for the agg_... tables
    names_in_schema = [
        "sites",
        "movies",
        "photos",
        "subjects",
        "species",
        "agg_annotations_clip",
        "sqlite_sequence",
        "agg_annotations_frame",
    ]
    assert (
        names == names_in_schema
    ), f"The list of table names is not as we would expect it from schema.py. We expect: {names_in_schema}, instead we get {names}"


def _get_row_count(conn: sqlite3.Connection, table):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    row_count = cur.fetchone()[0]
    return row_count


def test_add_to_table(connection):
    # If create_db is successfull, the db has a table called "sites"
    table = "sites"
    start_row_count = _get_row_count(connection, table)
    # Add one row of data to this table, check if it exist.
    db_utils.add_to_table(
        connection,
        table,
        [(1, "a", 10, 10, 10, 10), (2, "b", 20, 20, 20, 20)],
    )
    after_row_count = _get_row_count(connection, table)
    assert after_row_count == (
        start_row_count + 2
    ), f"_insert_many did not succeed to insert 2 rows, the row count before and after the function are {start_row_count} and {after_row_count}"
    # Test if we get an error telling us that the amount of columns are wrong when data is specified wrongly
    with pytest.raises(sqlite3.Error, match="columns but"):
        db_utils.add_to_table(
            connection,
            table,
            [(1, "a", 10, 10, 10), (2, "b", 20, 20, 20)],
        )


def test_empty_table(connection):
    # If create_db is successfull, the db has a table called "sites"
    table = "sites"
    # Add one row of data to this table, check if it exist.
    db_utils.add_to_table(connection, table, [(3, "c", 10, 10, 10, 10)])
    row_count = _get_row_count(connection, table)
    assert (
        row_count != 0
    ), f"It failed to add a row of data to the table '{table}', making it impossible to test the function empty_table. Look more at the test of _insert_many"

    # Then check if the empty_table function removes all data
    db_utils.empty_table(connection, table)
    empty_row_count = _get_row_count(connection, table)
    assert (
        empty_row_count == 0
    ), f"empty_table did not remove all data, the row count is {empty_row_count}"

    # Check if the table is still exists
    table_names = db_utils.get_schema_table_names(connection)
    assert (
        table in table_names
    ), f"the table '{table}' does not exist anymore. empty_table should only empty the table, not delete it"

    # Also test if we get an error when we try to remove a table that does not exist
    with pytest.raises(sqlite3.Error, match="does not exist"):
        db_utils.empty_table(connection, "random")


def test_rename_to_schema():
    template_project = project_utils.find_project("Template project")
    # Test if the function fails when no df is entered
    with pytest.raises(TypeError, match="Expected a pandas Dataframe"):
        db_utils.cols_rename_to_schema(template_project, "species", "fail")

    # Function works for project without renaming, example = template project
    df_template_no_col_rename = pd.DataFrame(
        [[1, "Nothing here", "Not applicable", "Not applicable", "Not applicable"]],
        columns=["species_id", "commonName", "scientificName", "taxonRank", "kingdom"],
    )
    df_template_renamed = db_utils.cols_rename_to_schema(
        template_project, "species", df_template_no_col_rename
    )
    assert list(df_template_no_col_rename.columns) == list(
        df_template_renamed.columns
    ), f"The template project does not need renaming, so it should return the same df. This is not the case. Original had column names {df_template_no_col_rename.columns}, returned df had {df_template_renamed.columns}"

    # Test if the renaming works for koster, taken as the exampel to test
    koster_project = project_utils.find_project("Koster_Seafloor_Obs")
    df_koster = pd.DataFrame(
        [
            [
                1,
                "movie_1.mp4",
                "Site_1",
                "13/08/2021",
                "Author_name_1",
                25.0,
                10.12,
                0.0,
                10.12,
                "https://www.wildlife.ai/wp-content/uploads/2022/06/movie_1.mp4",
            ],
            [
                2,
                "movie_2.mp4",
                "Site_1",
                "13/08/2021",
                "Author_name_2",
                29.97002997002997,
                10.043366666666667,
                0.0,
                10.043366666666667,
                "https://www.wildlife.ai/wp-content/uploads/2022/06/movie_2.mp4",
            ],
        ],
        columns=[
            "movie_id",
            "filename",
            "siteName",
            "created_on",
            "author",
            "fps",
            "duration",
            "SamplingStart",
            "SamplingEnd",
            "fpath",
        ],
    )
    # First if it works for a table that does not need any renaming within koster
    df_koster_renamed = db_utils.cols_rename_to_schema(
        koster_project, "species", df_koster
    )
    assert list(df_koster.columns) == list(
        df_koster_renamed.columns
    ), f"The species table in the koster project does not need renaming, so it should return the same df. This is not the case. Original had column names {df_koster.columns}, returned df had {df_koster_renamed.columns}"
    # Now with the movies table that does get renaming
    df_koster_renamed = db_utils.cols_rename_to_schema(
        koster_project, "movies", df_koster
    )
    assert list(df_koster.columns) != list(
        df_koster_renamed.columns
    ), f"The movies table in koster needs renaming, so it should not be the same as the original. But it is."
    assert (
        "sampling_start" in df_koster_renamed.columns
    ), f"If the renaming was successfull, 'sampling_start' should have replaced 'SamplingStart'. But it did not, the column names of the renamed df are {df_koster_renamed.columns}"


def test_process_test_csv(connection):
    template_project = project_utils.find_project("Template project")

    # First test if a correct df passes
    df_template = pd.DataFrame(
        [
            [1, "Nothing here", "Not applicable", "Not applicable", "Not applicable"],
            [2, "Banded weedfish", "Ericentrus rubrus", "species", "Animalia"],
            [3, "Banded wrasse", "Notolabrus fucicola", "species", "Animalia"],
        ],
        columns=["species_id", "commonName", "scientificName", "taxonRank", "kingdom"],
    )
    db_utils.process_test_csv(connection, template_project, df_template, "species")

    # Test if error gets raised when the cols do not match db schema
    df_wrong_col = pd.DataFrame(
        [
            [1, "Nothing here", "Not applicable", "Not applicable", "Not applicable"],
            [2, "Banded weedfish", "Ericentrus rubrus", "species", "Animalia"],
            [3, "Banded wrasse", "Notolabrus fucicola", "species", "Animalia"],
        ],
        columns=[
            "species_id",
            "common_Name",
            "scientific_Name",
            "taxonRank",
            "kingdom",
        ],
    )
    with pytest.raises(AssertionError, match="csv columns and db table columns for"):
        db_utils.process_test_csv(connection, template_project, df_wrong_col, "species")

    # Test spyfish movies
    spyfish_project = project_utils.find_project("Spyfish_Aotearoa")
    df_movies_valueerror = pd.DataFrame(
        [
            [
                1,
                "movie_1.mp4",
                "Site_1",
                "13/08/2021",
                "Author_name_1",
                25.0,
                10.12,
                0.0,
                10.12,
                "https://www.wildlife.ai/wp-content/uploads/2022/06/movie_1.mp4",
                None,  # This should give the value error
            ],
            [
                2,
                "movie_2.mp4",
                "Site_1",
                "13/08/2021",
                "Author_name_2",
                29.97002997002997,
                10.043366666666667,
                0.0,
                10.043366666666667,
                "https://www.wildlife.ai/wp-content/uploads/2022/06/movie_2.mp4",
                True,
            ],
        ],
        columns=[
            "movie_id",
            "filename",
            "siteName",
            "created_on",
            "author",
            "fps",
            "duration",
            "sampling_start",
            "sampling_end",
            "fpath",
            "IsBadDeployment",
        ],
    )
    with pytest.raises(ValueError, match="The 'IsBadDeployment' column"):
        db_utils.process_test_csv(
            connection, spyfish_project, df_movies_valueerror, "movies"
        )

    df_movies_correct = pd.DataFrame(
        [
            [
                1,
                "movie_1.mp4",
                "Site_1",
                "13/08/2021",
                "Author_name_1",
                25.0,
                10.12,
                0.0,
                10.12,
                "https://www.wildlife.ai/wp-content/uploads/2022/06/movie_1.mp4",
                False,
            ],
            [
                2,
                "movie_2.mp4",
                "Site_1",
                "13/08/2021",
                "Author_name_2",
                29.97002997002997,
                10.043366666666667,
                0.0,
                10.043366666666667,
                "https://www.wildlife.ai/wp-content/uploads/2022/06/movie_2.mp4",
                True,
            ],
        ],
        columns=[
            "movie_id",
            "filename",
            "siteName",
            "created_on",
            "author",
            "fps",
            "duration",
            "sampling_start",
            "sampling_end",
            "fpath",
            "IsBadDeployment",
        ],
    )
    df = db_utils.process_test_csv(
        connection, spyfish_project, df_movies_correct, "movies"
    )
    assert (
        len(df) == 1
    ), f"Expected to keep 1 row of data in spyfish, instead got {len(df)}"

    df_missing_isbaddeployment = pd.DataFrame(
        [[1, "movie_1.mp4"], [2, "movie_2.mp4"]], columns=["movie_id", "filename"]
    )
    with pytest.raises(
        AssertionError, match="The movies csv from Spyfish_Aotearoa expects"
    ):
        db_utils.process_test_csv(
            connection, spyfish_project, df_missing_isbaddeployment, "movies"
        )


def test_test_table_for_none():
    df_missing_id = pd.DataFrame(
        [
            [1, "Nothing here", "Not applicable", "Not applicable", "Not applicable"],
            [2, "Banded weedfish", "Ericentrus rubrus", "species", "Animalia"],
            [None, "Banded wrasse", "Notolabrus fucicola", "species", "Animalia"],
        ],
        columns=["species_id", "commonName", "scientificName", "taxonRank", "kingdom"],
    )
    with pytest.raises(AssertionError, match="has invalid entries"):
        db_utils.test_table_for_none(df_missing_id, "species", ["species_id"])


# populate_db calls upon process_test_csv and add_to_table which are both already tested, or their building blocks.
