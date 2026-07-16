class FeatureManager:
    TARGET_COLUMN = "ride_requests"

    ENTITY_COLUMN = "zone_timestamp_key"

    TIMESTAMP_COLUMN = "timestamp"

    IDENTIFIER_COLUMNS= [
        ENTITY_COLUMN,
        "zone_id"
    ]

    HIGH_CARDINALITY_COLUMNS = [
       "zone_id"
    ]

    LOW_CARDINALITY_COLUMNS = [
       "zone_type",
       "borough"
    ]