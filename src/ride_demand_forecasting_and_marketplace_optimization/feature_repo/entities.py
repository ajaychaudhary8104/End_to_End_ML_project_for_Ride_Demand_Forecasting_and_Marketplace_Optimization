from feast import Entity, ValueType

zone_timestamp = Entity(
    name="zone_timestamp",
    join_keys=["zone_timestamp_key"],
    value_type=ValueType.STRING
)