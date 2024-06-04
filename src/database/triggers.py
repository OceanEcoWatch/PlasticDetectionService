from sqlalchemy import text

from src.database.connect import create_db_session


def unique_image_func():
    return text(
        """
        CREATE OR REPLACE FUNCTION prevent_duplicate_image_insert()
        RETURNS TRIGGER AS $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM images
                WHERE timestamp = NEW.timestamp
                AND ST_Equals(bbox, NEW.bbox)
                AND image_id = NEW.image_id
                AND job_id = NEW.job_id
            ) THEN
                RAISE EXCEPTION 'Image with the same timestamp, bbox, sentinel_hub_id and job_id already exists';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def unique_image_trigger():
    return text(
        """
        CREATE TRIGGER prevent_duplicate_image_insert
        BEFORE INSERT ON images
        FOR EACH ROW
        EXECUTE FUNCTION prevent_duplicate_image_insert();
        """
    )


def create_triggers():
    with create_db_session() as db_session:
        db_session.execute(unique_image_func())
        db_session.execute(unique_image_trigger())
        db_session.commit()
        print("Triggers created")
